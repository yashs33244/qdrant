use std::assert_matches;
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use fs_err as fs;

use super::{
    BLOCK_SIZE, DiskCache, DiskCacheConfig, DiskCacheFs, DiskCacheFsContext, DiskCacheRemote,
};
use crate::generic_consts::Sequential;
use crate::mmap::AdviceSetting;
use crate::universal_io::{
    OpenOptions, Populate, ReadRange, UniversalAppend, UniversalFlush, UniversalRead,
    UniversalReadFileOps, UniversalReadFs,
};

fn make_test_data(n_bytes: usize) -> Vec<u8> {
    (0..n_bytes).map(|i| (i % 251) as u8).collect()
}

struct Scenario {
    _tmp: tempfile::TempDir,
    remote_path: PathBuf,
    data: Vec<u8>,
    config: Arc<DiskCacheConfig>,
}

impl Scenario {
    fn new(n_bytes: usize) -> Self {
        let tmp = tempfile::Builder::new()
            .prefix("disk_cache_tests")
            .tempdir()
            .unwrap();
        let remote_dir = tmp.path().join("remote");
        let local_dir = tmp.path().join("local");
        fs::create_dir_all(&remote_dir).unwrap();
        fs::create_dir_all(&local_dir).unwrap();

        let remote_path = remote_dir.join("data.bin");
        let data = make_test_data(n_bytes);
        fs::write(&remote_path, &data).unwrap();

        Self {
            _tmp: tmp,
            remote_path,
            data,
            config: Arc::new(DiskCacheConfig::new(remote_dir, local_dir).unwrap()),
        }
    }

    fn expected_local_path(&self) -> PathBuf {
        self.config.local_path_for(&self.remote_path).unwrap()
    }

    fn open<R>(&self, prefill: bool) -> DiskCache<R>
    where
        R: DiskCacheRemote,
        <R::Fs as UniversalReadFileOps>::ContextConfig: Default,
    {
        self.open_with(prefill, false)
    }

    fn open_writeable<R>(&self, prefill: bool) -> DiskCache<R>
    where
        R: DiskCacheRemote,
        <R::Fs as UniversalReadFileOps>::ContextConfig: Default,
    {
        self.open_with(prefill, true)
    }

    fn open_with<R>(&self, prefill: bool, writeable: bool) -> DiskCache<R>
    where
        R: DiskCacheRemote,
        <R::Fs as UniversalReadFileOps>::ContextConfig: Default,
    {
        let populate = if prefill {
            Populate::PreferBackground
        } else {
            Populate::No
        };

        let fs = DiskCacheFs::<R>::from_context(DiskCacheFsContext {
            config: self.config.clone(),
            remote: Default::default(),
        })
        .unwrap();
        fs.open(
            &self.remote_path,
            OpenOptions {
                writeable,
                populate,
                need_sequential: false,
                advice: AdviceSetting::Global,
            },
            Default::default(),
        )
        .unwrap()
    }

    /// Append `additional_bytes` bytes to the remote file in-place.
    /// Returns the full new remote contents.
    fn grow_remote(&mut self, additional_bytes: usize) -> Vec<u8> {
        use std::io::Write;

        let old_len = self.data.len();
        let new_data = make_test_data(old_len + additional_bytes);
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&self.remote_path)
            .unwrap();
        file.write_all(&new_data[old_len..]).unwrap();
        self.data = new_data.clone();
        new_data
    }

    /// Overwrite the remote file's content in place (same length) so any read
    /// that hits the remote instead of the local mirror becomes visible.
    fn mangle_remote(&self) {
        use std::io::{Seek, SeekFrom, Write};

        let len = fs::metadata(&self.remote_path).unwrap().len();
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(&self.remote_path)
            .unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&vec![0xEE; len as usize]).unwrap();
    }
}

#[duplicate::duplicate_item(
    tests_mod       R               cfg_predicate               _PREFILL;
    [tests_prefill] [MmapFile]      [cfg(all())]                [true];
    [tests_mmap]    [MmapFile]      [cfg(all())]                [false];
    [tests_uring]   [IoUringFile]   [cfg(target_os = "linux")]  [false];
)]
#[cfg_predicate]
#[cfg(test)]
mod tests_mod {
    use std::sync::atomic::Ordering;

    use super::*;
    #[cfg_predicate]
    use crate::universal_io::R;

    const PREFILL: bool = _PREFILL;

    #[test]
    fn basic_read_returns_remote_bytes() {
        let scn = Scenario::new(BLOCK_SIZE * 3 + 100);
        let file = scn.open::<R>(PREFILL);

        // Read inside the first block.
        let bytes = file
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 10,
                length: 20,
            })
            .unwrap();
        assert_eq!(&*bytes, &scn.data[10..30]);

        // Last block includes the 100-byte tail.
        let last = scn.data.len() as u64;
        let bytes = file
            .read::<Sequential, u8>(ReadRange {
                byte_offset: last - 50,
                length: 50,
            })
            .unwrap();
        assert_eq!(&*bytes, &scn.data[scn.data.len() - 50..]);
    }

    /// `read_whole` on a zero-length remote must return an empty slice, not
    /// panic. The whole-object prefill schedules nothing for an empty file, so
    /// `init_from_open_prefill` resolves to `None` and falls back to a
    /// zero-length mirror (on io_uring `schedule_whole` returns without
    /// scheduling; on mmap it yields an empty read).
    #[test]
    fn read_whole_empty_remote_returns_empty() {
        let scn = Scenario::new(0);
        let file = scn.open::<R>(PREFILL);

        let bytes = file.read_whole::<u8>().unwrap();
        assert!(bytes.is_empty());
        assert_eq!(file.len::<u8>().unwrap(), 0);
    }

    #[test]
    fn read_spanning_multiple_blocks_is_contiguous() {
        let scn = Scenario::new(BLOCK_SIZE * 3 + 100);
        let file = scn.open::<R>(PREFILL);

        let start = (BLOCK_SIZE - 50) as u64;
        let len = (BLOCK_SIZE + 100) as u64;
        let bytes = file
            .read::<Sequential, u8>(ReadRange {
                byte_offset: start,
                length: len,
            })
            .unwrap();
        let start = start as usize;
        let end = start + len as usize;
        assert_matches!(bytes, Cow::Borrowed(_));
        assert_eq!(bytes.as_ref(), &scn.data[start..end]);
    }

    #[test]
    fn local_file_is_created_on_first_read() {
        let scn = Scenario::new(BLOCK_SIZE * 2);
        let expected_local = scn.expected_local_path();

        let file = scn.open::<R>(PREFILL);

        // Before the first read, the local file doesn't exist yet.
        assert!(
            !expected_local.exists(),
            "local file should not exist before first read: {}",
            expected_local.display(),
        );

        // Trigger one read. This must bring up the local file.
        let _ = file
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: 1,
            })
            .unwrap();

        assert!(
            expected_local.exists(),
            "local file should exist after first read"
        );
        assert_eq!(
            fs::metadata(&expected_local).unwrap().len(),
            scn.data.len() as u64,
            "local file should be sized to the remote",
        );
    }

    #[test]
    fn populate_fetches_every_block() {
        let scn = Scenario::new(BLOCK_SIZE * 3 + 100);
        let file = scn.open::<R>(PREFILL);

        file.populate().unwrap();

        let bytes = file
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: scn.data.len() as u64,
            })
            .unwrap();
        assert_eq!(&*bytes, &scn.data[..]);
    }

    #[test]
    fn read_past_end_returns_out_of_bounds() {
        let scn = Scenario::new(1024);
        let file = scn.open::<R>(PREFILL);

        let err = file
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 1000,
                length: 100,
            })
            .unwrap_err();
        assert_matches!(
            err,
            crate::universal_io::UniversalIoError::OutOfBounds { .. },
        );
    }

    /// Reopen with no prior reads leaves the local mirror untouched.
    #[test]
    fn reopen_without_prior_reads_keeps_local_uninitialized() {
        let scn = Scenario::new(BLOCK_SIZE * 2);
        let mut cache = scn.open::<R>(PREFILL);
        let expected_local = scn.expected_local_path();
        assert!(!expected_local.exists());

        cache.reopen().unwrap();

        // it it was scheduled for prefill, it materializes the local file
        if PREFILL {
            assert!(expected_local.exists());
        } else {
            assert!(!expected_local.exists());
        }

        // In both Populate::No and Populate::PreferBackground, we still
        // have local marked as uninitialized at this point
        assert!(!cache.is_ready());
    }

    /// Reopen on an unchanged remote must not resize, repopulate, or mutate
    /// the fetched bitmap.
    #[test]
    fn reopen_no_growth_does_not_repopulate() {
        let scn = Scenario::new(BLOCK_SIZE * 3);
        let mut cache = scn.open::<R>(PREFILL);

        let _ = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: 1,
            })
            .unwrap();

        let (len_before, populated_before, fetched_before) = {
            let local = cache.state().expect("local initialized after read").local;
            (
                local.mmap().len::<u8>().unwrap(),
                local.fully_populated.load(Ordering::Acquire),
                local.fetched.lock().clone(),
            )
        };

        cache.reopen().unwrap();

        let local = if PREFILL {
            // in case of Populate::PreferBackground, we need to await for
            // completion to get the local_state back.
            cache.state().unwrap().local
        } else {
            // in case of Populate::No, local_state should still be there
            // without forcing (re)initialization.
            assert!(cache.is_ready(), "local must still be initialized");
            cache.state().unwrap().local
        };

        assert_eq!(local.mmap().len::<u8>().unwrap(), len_before);
        assert_eq!(
            local.fully_populated.load(Ordering::Acquire),
            populated_before,
        );
        assert_eq!(local.fetched.lock().clone(), fetched_before);
    }

    /// Reads into the new section must fail before reopen (local mirror is at
    /// the old length) and succeed after reopen.
    #[test]
    fn reopen_growth_visible_after_reopen() {
        let mut scn = Scenario::new(BLOCK_SIZE * 2);
        let mut cache = scn.open::<R>(PREFILL);

        let original_len = scn.data.len() as u64;

        let _ = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: 1,
            })
            .unwrap();

        let new_data = scn.grow_remote(BLOCK_SIZE);

        let err = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: original_len,
                length: BLOCK_SIZE as u64,
            })
            .unwrap_err();
        assert_matches!(
            err,
            crate::universal_io::UniversalIoError::OutOfBounds { .. },
        );

        cache.reopen().unwrap();

        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: original_len,
                length: BLOCK_SIZE as u64,
            })
            .unwrap();
        assert_eq!(&*bytes, &new_data[original_len as usize..]);
    }

    /// When the remote grows and the original tail block was only partially
    /// populated, reopen must invalidate that block so the next read re-fetches
    /// it instead of returning the zero-filled bytes left by `set_len`.
    #[test]
    fn reopen_growth_refetches_partial_tail_block() {
        // Non-block-aligned remote: block 1 holds only 100 real bytes.
        let mut scn = Scenario::new(BLOCK_SIZE + 100);
        let mut cache = scn.open::<R>(PREFILL);

        // Touch the partial tail so block 1 ends up in the `fetched` bitmap
        // (its fetch is clamped to the old EOF).
        let _ = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: BLOCK_SIZE as u64,
                length: 1,
            })
            .unwrap();

        // Grow remote past the old tail block boundary.
        let new_data = scn.grow_remote(BLOCK_SIZE);

        cache.reopen().unwrap();

        // Read covers both the originally-partial range [BLOCK_SIZE..old_len)
        // and the newly-grown tail [old_len..BLOCK_SIZE*2). Without the
        // invalidation, the second half would be zeros from `set_len`.
        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: BLOCK_SIZE as u64,
                length: BLOCK_SIZE as u64,
            })
            .unwrap();
        assert_eq!(&*bytes, &new_data[BLOCK_SIZE..BLOCK_SIZE * 2]);
    }

    /// Append writes through to both the remote (the single grow+write op)
    /// and the local mirror: afterwards every byte — old and appended — is
    /// served locally, which mangling the remote makes observable.
    #[test]
    fn append_grows_remote_and_serves_tail_from_mirror() {
        use std::sync::atomic::Ordering;

        // Non-block-aligned start so the append begins mid-block.
        let scn = Scenario::new(BLOCK_SIZE + 100);
        let mut cache = scn.open_writeable::<R>(PREFILL);

        // Fetch everything so the partial tail block is in the mirror.
        cache.populate().unwrap();

        // The appended range crosses a block boundary.
        let appended: Vec<u8> = (0..BLOCK_SIZE).map(|i| (i % 251) as u8 ^ 0xAA).collect();
        let offset = cache.append(&appended).unwrap();
        assert_eq!(offset, (BLOCK_SIZE + 100) as u64);
        assert_eq!(cache.len::<u8>().unwrap(), (2 * BLOCK_SIZE + 100) as u64);

        // The remote file received the appended bytes.
        let mut expected = scn.data.clone();
        expected.extend_from_slice(&appended);
        assert_eq!(fs::read(&scn.remote_path).unwrap(), expected);

        // Write-through proof: reads must be served from the mirror, so
        // mangling the remote must not show through.
        scn.mangle_remote();
        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: expected.len() as u64,
            })
            .unwrap();
        assert_eq!(&*bytes, &expected[..]);

        // populate + write-through covers every block.
        let local = cache.state().unwrap().local;
        assert!(local.fully_populated.load(Ordering::Acquire));
    }

    /// Appending while the pre-append tail block was never fetched: the
    /// block stays unfetched (its prefix is not locally valid), and reads of
    /// both the old and the appended range remain correct via lazy fetch.
    #[test]
    fn append_into_unfetched_tail_block_stays_consistent() {
        let scn = Scenario::new(BLOCK_SIZE + 100);
        let mut cache = scn.open_writeable::<R>(PREFILL);

        // Touch only block 0 — without prefill, the partial tail block 1
        // stays unfetched.
        let _ = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: 1,
            })
            .unwrap();

        // Stays within block 1.
        let appended = vec![0xCD_u8; 50];
        let offset = cache.append(&appended).unwrap();
        assert_eq!(offset, (BLOCK_SIZE + 100) as u64);

        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: BLOCK_SIZE as u64,
                length: 150,
            })
            .unwrap();
        let mut expected = scn.data[BLOCK_SIZE..].to_vec();
        expected.extend_from_slice(&appended);
        assert_eq!(&*bytes, &expected[..]);
    }

    /// Appending without any prior read materializes the mirror first.
    #[test]
    fn append_materializes_uninitialized_cache() {
        let scn = Scenario::new(100);
        let mut cache = scn.open_writeable::<R>(PREFILL);

        let offset = cache.append(b"tail".as_slice()).unwrap();
        assert_eq!(offset, 100);
        assert_eq!(cache.len::<u8>().unwrap(), 104);

        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 100,
                length: 4,
            })
            .unwrap();
        assert_eq!(&*bytes, b"tail");
    }

    #[test]
    fn append_batch_lands_contiguously_and_empty_append_is_noop() {
        let scn = Scenario::new(10);
        let mut cache = scn.open_writeable::<R>(PREFILL);

        // The flusher must succeed before materialization too.
        (cache.flusher())().unwrap();

        let batch: [&[u8]; 3] = [b"ab", b"", b"cde"];
        assert_eq!(cache.append_batch(batch).unwrap(), 10);
        assert_eq!(cache.append::<u8>(&[]).unwrap(), 15);
        assert_eq!(cache.len::<u8>().unwrap(), 15);

        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 10,
                length: 5,
            })
            .unwrap();
        assert_eq!(&*bytes, b"abcde");

        // After appends the flusher delegates to the remote's flusher.
        (cache.flusher())().unwrap();
    }

    #[test]
    fn append_requires_writeable_open() {
        let scn = Scenario::new(10);
        let mut cache = scn.open::<R>(PREFILL);

        let err = cache.append(b"x".as_slice()).unwrap_err();
        assert_matches!(err, crate::universal_io::UniversalIoError::Io(_));
    }

    /// Foreign growth between materialization and append: the mirror is
    /// stale, so append takes the resize-only path and lazy fetches heal the
    /// gap on the next read.
    #[test]
    fn append_after_foreign_growth_heals_lazily() {
        let mut scn = Scenario::new(100);
        let mut cache = scn.open_writeable::<R>(PREFILL);

        let _ = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 0,
                length: 1,
            })
            .unwrap();

        // The remote grows behind the cache's back...
        let grown = scn.grow_remote(50);

        // ...so this append lands at 150, not at the mirror's 100.
        let offset = cache.append(b"tail".as_slice()).unwrap();
        assert_eq!(offset, 150);
        assert_eq!(cache.len::<u8>().unwrap(), 154);

        // Both the foreign bytes and the appended bytes read correctly.
        let bytes = cache
            .read::<Sequential, u8>(ReadRange {
                byte_offset: 100,
                length: 54,
            })
            .unwrap();
        let mut expected = grown[100..].to_vec();
        expected.extend_from_slice(b"tail");
        assert_eq!(&*bytes, &expected[..]);
    }
}
