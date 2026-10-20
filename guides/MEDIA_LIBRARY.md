# Media Library / メディアライブラリ

![DarkモードのLibraryで検索した画像と詳細を確認 / Library image selection and details](images/media-library-dark-ja.jpg)

画面例は日本語表示です。 / Screenshots show the Japanese interface.

## 日本語

Media Libraryでは、画像、動画、音声を一つの画面で管理できます。上部の**すべて／画像／動画／音声**で種類を絞り込めます。

### カードと詳細

- 画像: プレビューと生成内容の詳細を表示します。
- 動画: ポスター画像、再生時間、解像度、音声の有無、状態を表示します。詳細ではプレーヤー、プロンプト、生成元、FPS、コーデック、生成設定、参照素材を確認できます。
- 音声: 再生時間、形式、サンプルレート、チャンネル、生成元とモデルを表示します。詳細では内蔵プレーヤー、説明、歌詞、シード、長さ、生成設定を確認できます。

動画と音声はカード一覧で自動再生されません。詳細のプレーヤーから再生します。元のファイルと、表示に使うプレビューやポスター画像は別々の成果物として保存されます。

### Video への受け渡し

- 画像の **Image to Video** は、選択した画像を Video Studio の入力へ渡します。
- 画像・動画・音声の **Reference / Video Edit** は、選択した素材を参照へ渡します。
- Sequenceの完成動画では、連結した動画と採用済みの各クリップを確認し、Sequenceへ戻れます。

選択した成果物はそのまま受け渡され、別のファイルへ勝手に置き換わりません。対応しない操作は、理由を表示したうえで無効になります。

### Music への受け渡し

生成音声で**音楽スタジオで再生成**を選ぶと、説明、歌詞、モード、シードなどの設定だけをMusic Studioへ渡します。音声ファイル自体は入力素材として使用しません。

画像専用の Image Edit、Topaz などの操作は、動画・音声を選択している間は利用できません。

### 部位を見比べる・生成条件を調べる

画像詳細の「人体部位レビュー」では、任意モデルが検出した矩形候補を修正・除外できます。Ctrl＋クリックなどで2〜32画像を選び「部位を比較」を開くと、同じ種類の部位を横並びで確認できます。候補は正常性や人物所属の判定ではありません。

「再生成アシスト」では設定を再利用し、「記録された数値の差」から保存済みの資産・実入力・sigma等を比較できます。未記録の値は推測しません。詳しい手順と任意モデルの準備は[制作支援ガイド](CREATIVE_WORKFLOWS.md)を参照してください。

## English

Media Library manages images, videos, and audio in one place. Filter the grid with **All / Images / Videos / Audio**.

### Cards and details

- Images keep the existing image preview and details.
- Video cards show a poster, duration, resolution, audio indicator, and status. Details include a player, prompt, provider, FPS, codec, generation settings, and references.
- Audio cards show duration, format, sample rate, channels, and provider/model. Details include a native player, description, lyrics, seed, duration, and generation settings.

Videos and audio are played from their detail players rather than auto-playing every card. Original media and any required preview/poster are kept as separate artifacts.

### Handoffs to Video

- **Image to Video** sends the selected image artifact to Video Studio.
- **Reference / Video Edit** sends the selected image, video, or audio artifact as a reference.
- Sequence details show the assembled result and accepted clips, with a route back to Sequence.

Asteria preserves the selected artifact identity and never silently substitutes another output. Unsupported actions stay disabled with a reason.

### Handoffs to Music

For generated audio, **Regenerate in Music** copies only settings such as description, lyrics, mode, and seed into a Music draft. The audio file itself is not used as a conditioning source.

Image-only actions such as Image Edit and Topaz are unavailable while a video or audio item is selected.

### Compare parts and inspect generation conditions

Body-part review in image details lets you correct or exclude optional-model detection candidates. Select 2–32 images with Ctrl+click and open Compare body parts to view the same part type side by side. Candidates do not judge anatomical correctness or person ownership.

Regeneration assist restores settings; Recorded numerical differences compares saved assets, actual inputs and sigma observations without inferring missing values. See [Creative workflows](CREATIVE_WORKFLOWS.md) for steps and optional-model preparation.
