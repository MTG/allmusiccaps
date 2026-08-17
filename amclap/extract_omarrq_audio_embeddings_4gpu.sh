CUDA_VISIBLE_DEVICES=0 python extract_omarrq_audio_embeddings.py ../cfg/config_clap_embedding_extraction.gin \
  --gin-binds='AudioEmbeddingDataModule.filelist_path = "../yt_filelist_aa"' &
CUDA_VISIBLE_DEVICES=1 python extract_omarrq_audio_embeddings.py ../cfg/config_clap_embedding_extraction.gin \
  --gin-binds='AudioEmbeddingDataModule.filelist_path = "../yt_filelist_ab"' >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=2 python extract_omarrq_audio_embeddings.py ../cfg/config_clap_embedding_extraction.gin \
  --gin-binds='AudioEmbeddingDataModule.filelist_path = "../yt_filelist_ac"' >/dev/null 2>&1 &
CUDA_VISIBLE_DEVICES=3 python extract_omarrq_audio_embeddings.py ../cfg/config_clap_embedding_extraction.gin \
  --gin-binds='AudioEmbeddingDataModule.filelist_path = "../yt_filelist_ad"' >/dev/null 2>&1
