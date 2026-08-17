"""Tests for multiview training support."""

import torch


class TestDummyDatasetMultiview:
    """Tests for DummyTextAudioDataset with multiview format."""

    def test_single_view(self):
        from amclap.data.dummy_text_audio import DummyTextAudioDataset

        ds = DummyTextAudioDataset(num_samples=10, n_text_views=1, n_audio_views=1)
        item = ds[0]
        assert len(item) == 3  # audio_views, text_views, mask
        audio_views, text_views, mask = item
        assert isinstance(audio_views, list)
        assert len(audio_views) == 1
        assert isinstance(audio_views[0], torch.Tensor)
        assert isinstance(text_views, list)
        assert len(text_views) == 1
        assert isinstance(text_views[0], str)

    def test_dual_audio_views(self):
        from amclap.data.dummy_text_audio import DummyTextAudioDataset

        ds = DummyTextAudioDataset(num_samples=10, n_text_views=1, n_audio_views=2)
        item = ds[0]
        audio_views, text_views, mask = item
        assert len(audio_views) == 2
        assert len(text_views) == 1

    def test_multiple_text_views(self):
        from amclap.data.dummy_text_audio import DummyTextAudioDataset

        ds = DummyTextAudioDataset(num_samples=10, n_text_views=3, n_audio_views=1)
        item = ds[0]
        audio_views, text_views, mask = item
        assert len(audio_views) == 1
        assert len(text_views) == 3
        for t in text_views:
            assert isinstance(t, str)

    def test_dual_both(self):
        from amclap.data.dummy_text_audio import DummyTextAudioDataset

        ds = DummyTextAudioDataset(num_samples=10, n_text_views=2, n_audio_views=2)
        item = ds[0]
        audio_views, text_views, mask = item
        assert len(audio_views) == 2
        assert len(text_views) == 2


class TestCollateWithSkip:
    """Tests for collate_with_skip with new multiview format."""

    def test_single_view(self):
        from amclap.data.data_utils import collate_with_skip

        batch = [
            [[torch.randn(10)], ["text_a"], None],
            [[torch.randn(10)], ["text_b"], None],
        ]
        audio_list, text_lists, mask = collate_with_skip(batch)
        assert len(audio_list) == 1
        assert audio_list[0].shape == (2, 10)
        assert len(text_lists) == 1
        assert text_lists[0] == ["text_a", "text_b"]

    def test_dual_audio_views(self):
        from amclap.data.data_utils import collate_with_skip

        batch = [
            [[torch.randn(10), torch.randn(10)], ["text_a"], None],
            [[torch.randn(10), torch.randn(10)], ["text_b"], None],
        ]
        audio_list, text_lists, mask = collate_with_skip(batch)
        assert len(audio_list) == 2
        assert audio_list[0].shape == (2, 10)
        assert audio_list[1].shape == (2, 10)
        assert len(text_lists) == 1

    def test_dual_both(self):
        from amclap.data.data_utils import collate_with_skip

        batch = [
            [[torch.randn(10), torch.randn(10)], ["t1a", "t2a"], None],
            [[torch.randn(10), torch.randn(10)], ["t1b", "t2b"], None],
        ]
        audio_list, text_lists, mask = collate_with_skip(batch)
        assert len(audio_list) == 2
        assert len(text_lists) == 2
        assert text_lists[0] == ["t1a", "t1b"]
        assert text_lists[1] == ["t2a", "t2b"]

    def test_skip_none_audio(self):
        from amclap.data.data_utils import collate_with_skip

        batch = [
            [[torch.randn(10)], ["text_a"], None],
            [[None], ["text_b"], None],
            [[torch.randn(10)], ["text_c"], None],
        ]
        audio_list, text_lists, mask = collate_with_skip(batch)
        assert audio_list[0].shape == (2, 10)
        assert text_lists[0] == ["text_a", "text_c"]

    def test_skip_none_text(self):
        from amclap.data.data_utils import collate_with_skip

        batch = [
            [[torch.randn(10)], ["text_a"], None],
            [[torch.randn(10)], [None], None],
            [[torch.randn(10)], ["text_c"], None],
        ]
        audio_list, text_lists, mask = collate_with_skip(batch)
        assert audio_list[0].shape == (2, 10)
        assert text_lists[0] == ["text_a", "text_c"]

    def test_with_tensor_mask(self):
        from amclap.data.data_utils import collate_with_skip

        batch = [
            [[torch.randn(10)], ["text_a"], torch.tensor([1.0])],
            [[torch.randn(10)], ["text_b"], torch.tensor([2.0])],
        ]
        audio_list, text_lists, mask = collate_with_skip(batch)
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (2, 1)


class TestCollateDataModuleIntegration:
    """Integration test: DummyTextAudioDataModule with collate."""

    def test_dataloader_output_format(self):
        from amclap.data.dummy_text_audio import DummyTextAudioDataModule

        dm = DummyTextAudioDataModule(
            batch_size=4,
            num_workers=0,
            n_text_views=2,
            n_audio_views=2,
            half_precision=False,
        )
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        audio_list, text_lists, mask = batch
        assert len(audio_list) == 2
        assert audio_list[0].shape[0] == 4
        assert len(text_lists) == 2
        assert len(text_lists[0]) == 4
        assert len(text_lists[1]) == 4
