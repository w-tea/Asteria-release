# 最初の生成と編集 / Quick start

[日本語](#日本語) | [English](#english) | [README](../README.md)

## 日本語

起動方法や必要なPC環境は[README](../README.md)を参照してください。このガイドは、モデルを用意してAsteriaを起動した後の操作を説明します。画面例はDarkモード・日本語表示です。表示するモデルや画像は環境によって異なります。

### 1. 表示を選ぶ

左の「設定」→「外観」で、ライト・ダーク・システムから選びます。変更はすぐに反映されます。表示言語は上部のEN／日本語から切り替えられます。

![外観設定でダークを選択した画面](images/appearance-dark-ja.jpg)

### 2. 一枚生成する

1. 左の「生成」を開き、モデルプロファイル、Checkpoint、必要なVAEを選びます。「モデルを検索」ではモデル名やフォルダー名で候補を絞り込めます。未導入の場合はモデルを追加するか、「設定」→「起動準備」を確認してください。
2. 描きたい内容とネガティブプロンプトを入力します。「項目別に入力」を選ぶと、ベース・キャラクター・詳細に分けて文章を整理できます。人物の位置を指定する機能ではありません。
3. 右の「基本設定」でサイズ、Samplingステップなどを選びます。通常はSeedを`-1`にして、毎回新しい値を使います。
4. 「生成」を押し、完成画像を確認します。中断したいときは「停止」を押します。自動リピートをONにした場合は、終了条件も確認してください。

![保存済みの完成画像を開いた生成画面。左がプロンプト、中央が画像、右が生成条件です](images/generate-dark-ja.jpg)

画面例は、保存済みの生成画像を履歴から読み込んだ状態です。再生成用のSeedや設定は履歴に残りますが、別の実行環境で画素まで同じになる保証はありません。詳しい比較や途中状態の保存は[制作支援ガイド](CREATIVE_WORKFLOWS.md)を参照してください。

### 3. メディアライブラリで選ぶ

「メディア」で画像・動画・音声を確認します。画像カードを選ぶと、右にプレビューと設定が表示されます。Ctrl＋クリックで複数選択し、整理や比較へ進めます。動画・音声は詳細のプレーヤーから再生します。

![検索した画像と選択画像の詳細を表示するメディアライブラリ](images/media-library-dark-ja.jpg)

「体の部位を検出・選択」と複数画像の部位比較では、検出候補を自分で確認できます。人体の正常性や部位の所属は判定しません。操作と任意モデルの用意は[制作支援ガイド](CREATIVE_WORKFLOWS.md)を確認してください。

### 4. 必要な範囲だけ編集する

メディアライブラリで画像を選び、「画像編集で開く」を押します。描き直したい範囲にマスクを描いてください。任意のSAMがある場合は、「体の部位を検出・選択」で作った実マスクも使えます。色付きの範囲を確認し、ブラシ・消しゴムで直してから「マスクを保存」で保存し、生成へ送ります。元画像は直接上書きしません。

![既存の画像編集でマスクを確認して生成へ送る画面](images/image-edit-mask-dark-ja.jpg)

「マスク修復」を選んだ状態です。「範囲を描き直す」の案内に従って、編集する範囲を確認してください。

### 5. 学習やMergeへ進む

LoRA学習では画像とキャプションを用意し、学習内容を確認してから開始します。キャラクター用キャプションは、先頭のトリガーを残し、それに覚えさせたい固定特徴を文章から外してください。AIの点検候補は自動適用されません。

![LoRA学習の入力と保存済み実行履歴を確認する画面](images/lora-training-dark-ja.jpg)

Mergeでは、上部のページ内移動から「モデル・計算方式」→「詳細レシピ」→「候補スタジオ」→「レシピを保存」の順に進めます。モデル名やフォルダー名で候補を絞り込み、設定を確認してレシピを保存してください。PCの通常幅では、右上の「マージを実行」は設定や履歴をスクロールしても表示されたままです。未保存の変更がある場合は、最後に保存したレシピを実行します。変更も使いたいときは、先に「変更を保存」を押してください。

プレビューや比較を使い、生成画像で結果を判断してください。以下は設定画面の例であり、Merge結果の品質を示す画像ではありません。

![Merge Studioでモデルと計算方法を選ぶ画面](images/merge-dark-ja.jpg)

### 6. 目的に合うガイドを開く

- [プロンプト・生成過程・部位レビュー・再生成](CREATIVE_WORKFLOWS.md)
- [完成画像のチェックと保存の扱い](IMAGE_CHECK.md)
- [画像・動画・音声の管理](MEDIA_LIBRARY.md)
- [動画の作成](VIDEO_STUDIO.md)
- [曲の作成](MUSIC_STUDIO.md)
- [拡張機能の導入と復旧](EXTENSIONS.md)

画像チェックは任意です。ONの場合は、すべての生成・補正を終えた完成画像を正式保存前に一回確認します。AIの見逃しや誤判定は残ります。終了するときは、ブラウザーを閉じるだけでなく`Stop-Asteria.cmd`を実行してください。

## English

See the [README](../README.md) for installation and hardware requirements. This guide starts after Asteria is running and you have compatible models. Screenshots show the Japanese interface in Dark mode; your models and images will differ.

### 1. Choose your display

Open Settings → Appearance and choose Light, Dark or System. The theme changes immediately. Use EN / 日本語 at the top to switch language.

![Dark selected in Appearance settings](images/appearance-dark-ja.jpg)

### 2. Generate one image

1. Open Generate and select the model profile, checkpoint and required VAE. Use Search models to filter by model or folder name. Add a compatible model or review Ready to start in Settings if needed.
2. Enter the positive and negative prompts. Sectioned input organizes text; it does not assign people to spatial regions.
3. Choose dimensions and sampling settings in Basic settings. Use Seed `-1` for a fresh value during ordinary generation.
4. Press Generate and inspect the completed image. Stop cancels an ongoing run. Review the stop conditions before enabling automatic repeat.

![A saved result open in Generate, with prompt input, the completed image and generation settings](images/generate-dark-ja.jpg)

The screenshot shows a saved generation restored from history. History retains the Seed and settings, but pixel-identical output across execution environments is not guaranteed. See [Creative workflows](CREATIVE_WORKFLOWS.md) for comparisons and saved intermediate states.

### 3. Choose media in Library

Open Media to browse images, video and audio. Select an image card to see its preview and settings. Ctrl+click selects multiple items for organization or comparison. Video and audio play from their detail players.

![Filtered Library images and the selected image details](images/media-library-dark-ja.jpg)

Detect and select body parts and same-part comparison show detection candidates for your inspection, not anatomical correctness or ownership judgments. See [Creative workflows](CREATIVE_WORKFLOWS.md) for their optional models and operations.

### 4. Edit a selected region

Select an image in Media Library, click Open in Image Edit, and draw a mask over the area to change. With optional SAM available, you can also use a real mask from Detect and select body parts. Inspect the colored region, refine it with the brush or eraser, use Save mask, then send it to Generate. The original image is not directly overwritten.

![Image Edit with a real mask and the handoff to Generate](images/image-edit-mask-dark-ja.jpg)

The screenshot shows **Inpaint mask** selected. Use **Redraw the masked area** to identify the task, then inspect the region before sending it to Generate.

### 5. Continue to training or Merge

Prepare images and captions in LoRA Training, inspect the configuration, then explicitly start training. For character captions, keep the trigger first and omit stable features that the trigger should recall. AI review suggestions do not apply themselves.

![LoRA Training inputs and previously saved run history](images/lora-training-dark-ja.jpg)

In Merge, use the navigation at the top to move through Models & method → Advanced Recipe → Candidate Studio → Save recipe. Filter models by model or folder name, review the settings, then save the recipe. At normal desktop widths, Run merge stays at the top right while you scroll the settings or history. With unsaved changes, the action runs the last saved recipe. Use Save changes first if you want those changes included.

Judge the result using generated images and comparisons. The screenshot shows configuration, not evidence of merge quality.

![Merge Studio model selection and calculation settings](images/merge-dark-ja.jpg)

### 6. Continue with the relevant guide

- [Prompts, generation stages, body parts and regeneration](CREATIVE_WORKFLOWS.md)
- [Completed-image checks and saving](IMAGE_CHECK.md)
- [Image, video and audio management](MEDIA_LIBRARY.md)
- [Video creation](VIDEO_STUDIO.md)
- [Music creation](MUSIC_STUDIO.md)
- [Extension installation and recovery](EXTENSIONS.md)

Image checking is optional. When enabled, it runs once after all generation and enhancement, before formally saving the completed image. AI can still miss defects or report false positives. Run `Stop-Asteria.cmd` when finished; closing a browser tab alone does not stop Asteria.
