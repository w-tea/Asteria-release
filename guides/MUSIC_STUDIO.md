# Music Studio / ミュージックスタジオ

## 日本語

Music Studioは、**MiniMax-Music3**で新しい曲を作る画面です。V1.2.0では**Vocal**と**Instrumental**の新規生成に対応します。カバー、リミックス、ステム分離、DAW連携、マスタリングは対象外です。

APIの選択、APIキー、認証情報、課金設定はありません。ローカル生成に失敗しても、有料APIへ自動で切り替わりません。既存の音声ファイルを生成条件として入力する機能もありません。

### 使い方

1. **Vocal** または **Instrumental** を選びます。
2. 曲の説明を入力します。ジャンル、雰囲気、楽器、テンポ、構成、声のイメージなどを自然文で指定できます。
3. Vocal では必要に応じて歌詞を入力します。
4. シード、長さ、言語を設定し、生成します。

MiniMax-Music3が未導入の場合は、セットアップカードで実行プロファイルを選び、必要な空き容量とライセンスを確認してからダウンロードします。プロファイルはPCのメモリ容量に合わせて選んでください。設定によって負荷が高くなる場合は、GPU、メモリ、空き容量の見積もりと警告を表示します。警告は実行を禁止するものではありません。内容を確認し、続行する場合は**このまま実行**を選びます。

生成後はその場で試聴し、保存するかMedia Libraryを開けます。失敗した処理は**再試行**を選ぶと、同じ設定でもう一度実行できます。

### Media Library からの再生成

生成音声で**音楽スタジオで再生成**を選ぶと、説明、歌詞、モード、シードなどの生成設定だけをMusic Studioの下書きへコピーします。元の音声ファイルは入力素材として送りません。

## English

Music Studio is an independent workspace for creating new music locally with **MiniMax-Music3**. V1.2.0 supports new **Vocal** and **Instrumental** generation. Cover, Remix, stem separation, DAW, and mastering are outside this scope.

There is no API selector, API key, credential storage, billing flow, or paid fallback. Audio files cannot be used as Music Studio conditioning sources.

### Basic flow

1. Choose **Vocal** or **Instrumental**.
2. Describe the song in natural language: genre, mood, instruments, tempo, structure, and voice intent are all useful.
3. Add lyrics when creating a Vocal track.
4. Set the seed, desired duration, and language intent, then generate.

If Local MiniMax-Music3 is not ready, choose an execution profile in the setup card and review the required space and license before downloading. Select a profile that fits the machine’s memory. GPU/VRAM, RAM, disk, and workload estimates may produce a `caution` or `high risk` advisory. This is not a hard block; review it and choose **Run anyway** when you want to proceed.

Preview the result in the built-in player, save it, or open Media Library. A failed job can be retried with **Retry**. A local failure never silently switches to an API provider.

### Regenerate from Media Library

**Regenerate in Music** copies request settings such as description, lyrics, mode, and seed into a Music Studio draft. It does not send the existing audio artifact as a conditioning source.
