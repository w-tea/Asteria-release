# Asteria V1.1.0 Windows Portable

[日本語](#日本語) | [English](#english)

## 更新履歴 / Release history

- **V1.1.0** — 永続操作、更新・復旧保護、外部モデルライブラリ、追加Merge方式、全画面テーマ刷新、画像metadata再利用、Discord生成、診断を追加。 / Added durable operations, safer update and recovery, external model libraries, new Merge modes, a full visual refresh, image-metadata reuse, Discord generation, and diagnostics.
- **V1.0.4** — Runtime Monitor、生成時間、Seed単独適用、LoRAサムネイルと生成出力を改善。 / Added Runtime Monitor and generation time, seed-only reuse, LoRA thumbnails, and generation-output fixes.
- **V1.0.3** — LibraryとImage Editの長期利用・データ保護、Generate／Merge XYZ、Topaz／Eagle連携、モデル参照と操作UIを強化。 / Strengthened long-term Library and Image Edit safety, Generate/Merge XYZ, Topaz/Eagle integration, model handling, and operation UI.
- **V1.0.2** — Generateのモデル一覧cache、Generation Preset、Prompt & Assets、画面遷移時のrefresh契約を整理。 / Refined Generate model caching, Generation Presets, Prompt & Assets, and navigation refresh behavior.
- **V1.0.1** — Merge XYZ、Windows linked model、Portable bootstrap／repairを強化。 / Added Merge XYZ and strengthened Windows linked-model and Portable bootstrap/repair behavior.
- **V1.0.0** — Windows Portable初回公開版。 / Initial Windows Portable release.

この一覧には、各バージョンで利用者が使える正式な変更だけを記載します。 / This list contains only supported user-facing changes for each version.

## 日本語

### 概要

Asteriaは、ローカル環境で動作するWindows向けAI画像制作アプリです。画像生成、画像編集、生成結果の比較・整理、LoRA学習、モデルマージを1つの画面から行います。初回起動時には、動作に必要な実行環境を自動で取得します。画像生成に使うCheckpointは同梱されていないため、任意のスターターパックを取得するか、手持ちのモデルを追加してください。

### 主な機能

- **画像生成と補正**: txt2img／img2imgに加え、Hires.fix、ControlNet、Detailer、Topaz Photo AI連携を利用します。
- **画像編集とライブラリ**: マスクや切り抜きを使った編集、生成履歴の閲覧、画像の比較、評価やタグによる整理を行います。
- **プロンプトと生成設定の再利用**: スタイル、ワイルドカード、Embedding、LoRA、生成プリセットに加え、PNG／JPEG／WebP画像からプロンプトや対応する生成設定を再利用します。
- **LoRA学習**: 学習用画像とキャプションを準備し、Asteriaから学習を実行して結果をモデル一覧へ登録します。
- **Merge Studio**: CheckpointやLoRAのマージ条件を比較し、確認後にマージモデルを書き出します。
- **表示テーマと進行状況**: Light／Dark／Systemテーマを切り替え、画面を移動しても処理状況を確認・取消・再試行できます。
- **外部モデルライブラリ**: 手持ちのAsteria／ComfyUI／Forge系モデル保管先を登録し、元ファイルを移動せずに利用できます。
- **Discord生成**: Generateで明示公開した設定だけを、許可したDiscordサーバーやチャンネルから利用できます。

### 更新履歴

#### V1.1.0

- 長時間処理を永続操作として管理し、画面を移動しても進行状況、取消、再試行、復旧結果を確認できるようにしました。
- 中断可能なダウンロード、schema migration、バックアップ／復元、署名付き管理対象更新にデータ保護と再開処理を追加しました。
- Light／Dark／Systemテーマと、全画面共通の処理状況表示を追加しました。
- 外部モデルライブラリを登録し、元ファイルを移動せずに安定したモデル参照として利用できるようにしました。
- Windows標準のファイル／フォルダ選択を追加しました。外部モデルライブラリはフォルダ構成とモデル件数を自動判定し、初回登録で内部形式や数値優先度を入力する必要がなくなりました。
- Merge StudioへSLERP、NuSLERP、Nearswapを追加し、Weight sum、MBW、Merge XYZで利用できるようにしました。
- Generate設定を明示公開して利用するDiscord生成を追加しました。許可範囲、進行表示、取消、混雑制御、配送失敗時の復旧案内に対応します。
- Settingsへ秘匿情報やprompt、絶対パス、プロセス詳細を含まない診断出力と、安全なランタイム復旧操作を追加しました。
- 診断の警告へ具体的な解決手順を追加し、Discord設定には各IDの取得場所と使い方を表示するようにしました。
- Lightテーマのガイド、詳細カード、XYZ、AIプロンプト支援などの動的画面を見直し、画面の再訪時には読込済みデータを再利用するようにしました。
- 画像ライブラリへ戻った時は、表示中の一覧を残したまま最新72件だけを確認するようにしました。新しい生成画像をすぐ確認でき、更新に失敗しても現在の一覧は消えません。
- Windows標準の参照画面を開いている間は、アプリ内の参照モーダルを重ねて表示しないようにしました。
- PNG／JPEG／WebPを選び、Asteria、A1111／Forge、一般的なComfyUIのmetadataを確認して再利用できるようにしました。画像のプロンプトだけ、現在のSeedを変えずに生成設定、画像入力、Seedだけをそれぞれ明示的に適用でき、適用だけで生成は開始しません。
- 全画面の配色と面の区切り、通常／実行／選択／削除操作の見分けやすさ、Light／Dark／Systemでの可読性を見直しました。既存の画面構成と操作順は維持しています。
- 閉じる操作を×アイコンへ統一し、Libraryのフォルダー列、画像一覧のスクロール境界、Image Edit／Library／Mergeの狭い画面での配置、画面下部の不要な空白を改善しました。

#### V1.0.4

- 手持ちのCheckpointやVAEを設定している場合に、スターターパック不足の警告が残る問題を修正しました。
- 生成画像とは別に、中間PNGが`outputs/`へ残る問題を修正しました。
- 生成結果と履歴で、画像の生成時間を確認できるようにしました。
- LoRAと同じ名前の画像をサムネイルとして表示できるようにしました。ジャンクションやシンボリックリンクで共有しているLoRAにも対応します。
- SettingsにRuntime Monitorを追加しました。次回起動時に別ウィンドウで生成ログを確認でき、Monitorを閉じてもAsteria本体は終了しません。
- 生成結果や履歴から、選択した画像のSeedだけを現在の生成設定へ適用できるようにしました。

### 必要最低限のスペックと動作環境

- 64-bit版Windows 10またはWindows 11
- x64 CPU
- CUDA対応のNVIDIA GPU
- NVIDIAドライバー 570.65以上
- インターネット接続
- 展開先への書き込み権限
- 初回取得、追加するモデル、生成画像を保存できる空き容量

AMD／Intel GPUとCPUのみの実行には対応していません。macOSとLinuxもV1.1.0では未対応です。

必要なGPUメモリとシステムメモリは、使用するモデル、生成解像度、LoRA学習などの処理によって変わります。高解像度生成や学習を行う場合は、より多くのメモリが必要です。

### インストール方法

1. 配布ZIPを、十分な空き容量がある書き込み可能なフォルダーへ完全に展開します。ZIP内から直接起動しないでください。
2. 展開先にある`Start-Asteria.cmd`をダブルクリックします。
3. 初回セットアップでは、Python、Node.js、ComfyUI、必要なパッケージ、カスタムノード、機能用データを取得して検証します。CMDウィンドウは、完了またはエラーが表示されるまで閉じないでください。
4. 必須セットアップが完了すると、Asteriaがブラウザーで開きます。
5. 画像生成モデルが必要な場合は、AsteriaのSettingsから「起動準備」を開きます。SDXL、Anima、両方、または「モデルなしで続行」を選んでください。

Windowsへ事前にインストールする必要があるのは、対応するNVIDIAドライバーです。CUDA Toolkit、Python、Node.js、ComfyUIを個別に導入する必要はありません。

初回取得には時間がかかり、機能用データだけでも数十GBになる場合があります。実際の取得量はセットアップ画面に表示されます。処理を中断した場合も、検証済みのファイルは次回の実行で再利用されます。

SDXL／Animaスターターパックは必須ではありません。CMDで行う必須セットアップには含まれず、Asteria起動後に選択した場合だけ取得されます。両方をスキップした場合は、手持ちのモデルを追加して使用します。Animaスターターパックの取得には、画面に表示される非商用ライセンスへの同意が必要です。

### 起動方法

通常は、展開先の次のファイルを実行して起動します。

`Start-Asteria.cmd`

2回目以降は取得済みコンポーネントを検証し、変更がなければダウンロードを省略します。ブラウザーのタブを閉じただけでは、Asteriaの処理は終了しません。

利用を終えるときは、次のファイルを実行します。Asteriaが起動したPython、Node.js、ComfyUIもこの操作で終了します。

`Stop-Asteria.cmd`

初回取得に失敗したときや、管理対象ファイルの不足・破損が疑われるときは、次のファイルを実行します。Asteriaを停止してから、必要なファイルを再検証し、不足分を取得します。

`Repair-Asteria.cmd`

### モデルの追加

手持ちのモデルは、展開先の`models/`へ追加します。SDXL／Animaスターターパックを取得しなくても、Asteriaが対応するモデルを利用できます。

次のフォルダーは初回起動時に自動作成されます。モデルの種類に合う場所へ配置してください。

- Checkpoint: `models/checkpoints/`または`models/diffusion_models/`
- VAE: `models/vae/`
- テキストエンコーダー: `models/text_encoders/`
- LoRA: `models/loras/`
- Embedding: `models/embeddings/`
- ControlNet: `models/controlnet/`
- ADetailer／検出モデル: `models/adetailer/`または`models/bbox/`
- 領域分割モデル: `models/segm/`
- Upscaler: `models/upscalers/`

`sdxl`や`anima`などのサブフォルダーは任意です。モデルを追加した後は、Settingsの「モデルアセット」でモデル一覧を更新してください。

SDXLスターターパックは、Checkpointを`models/checkpoints/sdxl/`、VAEを`models/vae/sdxl/`へ配置します。Animaスターターパックは、モデルを`models/checkpoints/anima/`、テキストエンコーダーを`models/text_encoders/anima/`、VAEを`models/vae/anima/`へ配置します。

#### 別の場所にあるモデルを共有する

Settingsの「外部モデルライブラリ」から、共有`models`フォルダや既存のAsteria／ComfyUI／Forge系モデル保管先を登録できます。`StableDiffusion`／`Stable-diffusion`、`Lora`、`Embeddings`、`adetailer`、`bbox`、`segm`などを自動判定します。`adetailer`と`bbox`はどちらも検出モデルとして、`segm`は領域分割モデルとして取得します。Asteriaは登録先のモデル本体を移動・削除しません。接続できないドライブやネットワーク先はオフラインとして扱われ、再接続後に再スキャンできます。

従来どおり、`models/`内のジャンクションまたはシンボリックリンクで共有することもできます。リンクを作る場合は、まず`Stop-Asteria.cmd`でAsteriaを終了してください。

一般的なWindows環境では、管理者権限を要求しにくいジャンクションを使用します。次のコマンドをコマンドプロンプトで実行します。

```bat
mklink /J "<Asteriaのモデル種別フォルダー内に作るリンク>" "<既存モデルフォルダー>"
```

たとえば、Checkpointの共有先として`models\checkpoints\my-library`を作成できます。`my-library`の名前は任意です。ディレクトリまたはファイルのシンボリックリンクも読み込めますが、WindowsのDeveloper Modeまたは管理者権限が必要になる場合があります。

リンク作成後はAsteriaを起動し、Settingsの「モデルアセット」で一覧を更新してください。リンク先に接続できない場合、そのモデルは利用できません。Asteriaと`Repair-Asteria.cmd`は、リンク先のモデルを移動、削除、修復しません。

`models/checkpoints`や`models/vae`など、スターターパックの取得先そのものをリンクにしないでください。モデルの読み込みはできても、安全上の理由からスターターパックの取得や修復が停止する場合があります。

### 主な保存場所

- `models/`: 手持ちのモデル、スターターパック、モデルへのリンク
- `data/`: 設定、生成履歴、プリセット、画像編集用データ、学習やマージの作業データ
- `outputs/`: 生成画像、編集結果、補正後の画像
- `logs/`: 起動処理、ComfyUI、Asteria本体の診断ログ

これらはすべてAsteriaの展開フォルダー内にあります。Asteriaの使用中に、`data/`や`outputs/`を手動で移動しないでください。

### Eagle連携

生成完了時に、最終生成画像をEagleへ自動送信できます。Eagleを起動してから、Settingsの「外部ツール」でEagleのURL、送信先フォルダー、「生成完了時にEagleへ自動送信」を設定して保存してください。

Hires.fixなどの中間画像は自動送信の対象外です。Eagleへの送信に失敗しても画像生成は完了します。接続を直した後、ライブラリの履歴から手動で再送できます。

### Discord生成

Settingsの「Discord」でBot接続先と許可範囲を保存した後、Generateの「Discordへ公開」で現在の設定を公開します。公開後にGenerateを編集してもDiscord側の設定は変わりません。変更を反映するには再公開してください。公開解除中はDiscordから生成できません。

Bot tokenは保存後に再表示されません。許可するサーバー、チャンネル、ユーザーを必要最小限にし、不要になった場合は公開解除とtoken削除を行ってください。

### 診断、バックアップ、復旧

Settingsの「診断」では、ランタイムの準備状況と必要な対応を確認できます。明示的に出力した診断レポートには、prompt、token、絶対パス、プロセス情報、raw logは含まれません。完了済み記録の整理と、Asteriaが所有権を確認できたランタイムの安全な再起動もここから行えます。

Settingsの「メンテナンス」では、互換性情報を確認し、バックアップを作成できます。復元時は内容の事前確認と明示確認が必要です。更新や復元に失敗した場合は、Asteriaを停止してから`Repair-Asteria.cmd`を実行してください。

### よくある質問（簡易Q&A）

#### Q. `Start-Asteria.cmd`を実行してもブラウザーが開きません。

まずCMDウィンドウの最後のメッセージを確認してください。初回セットアップ中であれば、そのまま完了まで待ちます。

エラーが表示された場合は、NVIDIAドライバー、空き容量、インターネット接続、セキュリティソフトによる遮断を確認してください。その後、`Repair-Asteria.cmd`を実行します。必須セットアップ後の起動に失敗した場合は、`logs/launcher/`のログも確認してください。

#### Q. 初回ダウンロードが長い、または途中で止まりました。

CMDに通信エラーなどが表示されていないか確認してください。中断した場合は、再度`Start-Asteria.cmd`を実行します。検証済みのファイルは再利用され、不足分から処理を続けます。

#### Q. 2回目以降もダウンロードされます。

不足、破損、または配布manifestの変更が検出されたコンポーネントだけを再取得します。毎回同じ項目で止まる場合は、空き容量とセキュリティソフトを確認してから`Repair-Asteria.cmd`を実行してください。

#### Q. 追加したモデルが一覧に表示されません。

モデルの種類に合うフォルダーへ配置されているか確認し、Settingsの「モデルアセット」で一覧を更新してください。リンクを使用している場合は、リンク先のドライブやネットワーク先へ接続できることも確認します。

#### Q. `A stale launcher record must be stopped or repaired before starting.`と表示されます。

`Stop-Asteria.cmd`を実行してから、もう一度`Start-Asteria.cmd`を実行してください。同じ表示が続く場合は`Repair-Asteria.cmd`を実行します。

#### Q. 展開したフォルダーを移動できますか。

`Stop-Asteria.cmd`を実行してAsteriaを完全に終了してから、フォルダー全体を移動してください。起動中には移動しないでください。

### バックアップとアンインストール

バックアップを作成するときは、先に`Stop-Asteria.cmd`を実行してください。少なくとも`data/`、`models/`、`outputs/`を保存します。問題調査に備えて診断記録も残す場合は、`logs/`も保存してください。

ジャンクションやシンボリックリンクを使用している場合、`models/`のバックアップだけではリンク先の実ファイルを保存できないことがあります。外部モデルの保管先も別途バックアップしてください。

アンインストールするときは、`Stop-Asteria.cmd`で終了した後、Asteriaの展開フォルダーを削除します。展開フォルダー内の設定、履歴、モデル、生成画像、ログも削除されるため、必要なデータを先にバックアップしてください。

### ライセンスと第三者コンポーネント

Asteria本体のライセンスは`LICENSE`に記載されています。

取得する実行環境、パッケージ、カスタムノード、モデルの提供元、固定バージョン、ライセンスは、`release-manifest.yaml`、`runtime-dependencies.yaml`、`model-dependencies.yaml`、`notices/THIRD_PARTY_NOTICES.txt`で確認できます。スターターパックを含む第三者モデルには、それぞれの配布元が定める利用条件が適用されます。

---

## English

### Overview

Asteria is a Windows application for local AI image production. It covers image generation, image editing, result comparison and organization, LoRA training, and model merging in one interface. On the first launch, Asteria downloads the runtime components required to operate. Generation checkpoints are not bundled, so install an optional starter pack or add your own models.

### Main features

- **Generation and enhancement**: Use txt2img, img2img, Hires.fix, ControlNet, Detailer, and Topaz Photo AI integration.
- **Image editing and Library**: Edit with masks and crops, browse generation history, compare images, and organize them with ratings and tags.
- **Reusable prompts and settings**: Reuse styles, wildcards, embeddings, LoRAs, generation presets, and compatible prompts or settings found in PNG, JPEG, and WebP images.
- **LoRA training**: Prepare training images and captions, run training from Asteria, and register the result in the model list.
- **Merge Studio**: Compare checkpoint or LoRA merge settings, then write the selected merged model.
- **Themes and operation status**: Select Light, Dark, or System and keep track of running work across workspace navigation, including safe cancel and retry.
- **External model libraries**: Register existing Asteria, ComfyUI, or Forge-style model locations without moving their source files.
- **Discord generation**: Use only the Generate settings you explicitly publish from allowed Discord servers or channels.

### Release history

#### V1.1.0

- Added durable operations so long-running work keeps its status across navigation and can be cancelled, retried, and recovered safely.
- Added resumable downloads, schema migrations, backup and restore safeguards, and signed managed-update protection.
- Added Light, Dark, and System themes plus shared operation progress across all workspaces.
- Added external model libraries with stable references while leaving source model files in place.
- Added SLERP, NuSLERP, and Nearswap to Merge Studio for Weight sum, MBW, and Merge XYZ workflows.
- Added Discord generation from explicitly published Generate settings, including allowlists, progress, cancellation, queue controls, and actionable delivery/runtime recovery.
- Added sanitized diagnostics and ownership-checked runtime recovery without exporting secrets, prompts, absolute paths, or process internals.
- Added concrete recovery steps to diagnostics warnings and in-place guidance showing where Discord IDs come from and how each one is used.
- Improved Light-theme readability for dynamic guides, detail cards, XYZ, and AI Prompt Assist, and reused loaded workspace data when revisiting screens.
- When returning to the Image Library, Asteria now checks only the latest 72 matching records while keeping the current list visible. New generations appear promptly, and a failed check no longer clears the list.
- Prevented the in-app path browser from appearing behind the native Windows picker.
- Added preview and reuse of Asteria, A1111/Forge, and common ComfyUI metadata from PNG, JPEG, and WebP images. Apply prompts only, compatible generation settings without changing the current Seed, the image input, or the Seed alone; applying settings never starts generation automatically.
- Improved visual hierarchy across every workspace, including clearer surfaces and normal, primary, selected, and destructive actions in Light, Dark, and System themes, while preserving the existing screen structure and workflow order.
- Standardized direct close actions as × icons and improved the Library folder rail and scrolling boundary, narrow Image Edit/Library/Merge layouts, and unintended empty space below workspaces.

#### V1.0.4

- Fixed the starter-pack warning remaining visible when your own checkpoint or VAE is already selected.
- Fixed intermediate PNG files being left under `outputs/` in addition to the final generated image.
- Added generation-time information to results and history.
- Added LoRA thumbnails from image files with the same name, including LoRAs shared through junctions or symbolic links.
- Added Runtime Monitor under Settings. It can open on the next launch to show generation logs in a separate window, and closing it does not stop Asteria.
- Added an action to apply only the seed from a selected result or history image to the current generation settings.

### Minimum requirements and supported environment

- 64-bit Windows 10 or Windows 11
- An x64 CPU
- A CUDA-capable NVIDIA GPU
- NVIDIA driver 570.65 or newer
- Internet access
- Write access to the extracted folder
- Enough free space for first-run downloads, added models, and generated images

AMD and Intel GPUs and CPU-only operation are unsupported. macOS and Linux are also unsupported in V1.1.0.

GPU-memory and system-memory requirements depend on the selected model, output resolution, and workloads such as LoRA training. High-resolution generation and training require more memory.

### Installation

1. Fully extract the distribution ZIP to a writable folder with enough free space. Do not run Asteria from inside the ZIP.
2. Double-click `Start-Asteria.cmd` in the extracted folder.
3. On the first run, setup downloads and verifies Python, Node.js, ComfyUI, required packages, custom nodes, and functional assets. Keep the CMD window open until it reports completion or an error.
4. Asteria opens in the browser after required setup finishes.
5. If you need a generation model, open **Ready to start** in Settings. Select SDXL, Anima, both, or **Continue without starter models**.

The only prerequisite that must be installed in Windows is a compatible NVIDIA driver. You do not need to install the CUDA Toolkit, Python, Node.js, or ComfyUI separately.

The first download takes time and may require tens of GB for functional assets alone. Setup displays the planned download size. If setup is interrupted, files that already passed verification are reused on the next run.

The SDXL and Anima starter packs are optional. They are not part of the required CMD setup and are downloaded only after you select them in Asteria. You may skip both and add your own models instead. Installing the Anima starter pack requires acceptance of the displayed non-commercial license.

### Starting Asteria

For a normal start, run the following file from the extracted folder.

`Start-Asteria.cmd`

On later starts, Asteria verifies previously acquired components and skips downloads when nothing has changed. Closing the browser tab does not stop Asteria processes.

When you finish using Asteria, run the following file. This also stops the Python, Node.js, and ComfyUI processes started by Asteria.

`Stop-Asteria.cmd`

If the first download failed or a managed component may be missing or corrupt, run the following file. It stops Asteria, verifies the managed files, and reacquires missing items.

`Repair-Asteria.cmd`

### Adding models

Add your own models under `models/` in the extracted folder. You may skip the SDXL and Anima starter packs and use any compatible models you already have.

The following folders are created automatically on first launch. Place each file in the folder for its model type.

- Checkpoint: `models/checkpoints/` or `models/diffusion_models/`
- VAE: `models/vae/`
- Text encoder: `models/text_encoders/`
- LoRA: `models/loras/`
- Embedding: `models/embeddings/`
- ControlNet: `models/controlnet/`
- ADetailer / detector: `models/adetailer/` or `models/bbox/`
- Segmentation detector: `models/segm/`
- Upscaler: `models/upscalers/`

Additional subfolders such as `sdxl` or `anima` are optional. After adding a model, refresh the model list under **Model Assets** in Settings.

The SDXL starter pack places its checkpoint in `models/checkpoints/sdxl/` and its VAE in `models/vae/sdxl/`. The Anima starter pack places its model in `models/checkpoints/anima/`, its text encoder in `models/text_encoders/anima/`, and its VAE in `models/vae/anima/`.

#### Sharing models stored elsewhere

Register a shared `models` folder or an existing Asteria, ComfyUI, or Forge-style model location under External Model Libraries in Settings. Asteria automatically recognizes folders such as `StableDiffusion` / `Stable-diffusion`, `Lora`, `Embeddings`, `adetailer`, `bbox`, and `segm`. Both `adetailer` and `bbox` are loaded as detector sources, while `segm` is loaded as a segmentation source. Asteria does not move or delete model files at the registered location. Disconnected drives and network locations are shown as offline and can be rescanned after reconnection.

You can still share models through a junction or symbolic link inside `models/`. Stop Asteria with `Stop-Asteria.cmd` before creating a link.

A junction is the usual choice on Windows because it is less likely to require administrator privileges. Run the following command in Command Prompt.

```bat
mklink /J "<link to create inside an Asteria model-type folder>" "<existing model folder>"
```

For example, you can create `models\checkpoints\my-library` for a checkpoint library. The name `my-library` is only an example. Directory and file symbolic links are also supported, but they may require Windows Developer Mode or administrator privileges.

After creating the link, start Asteria and refresh the list under **Model Assets** in Settings. A model becomes unavailable when its link target cannot be reached. Asteria and `Repair-Asteria.cmd` do not move, delete, or repair files at the link target.

Do not replace a managed destination such as `models/checkpoints` or `models/vae` itself with a link. Reading may still work, but starter-pack installation or repair may stop to protect the linked files.

### Main storage locations

- `models/`: Your models, starter packs, and links to model libraries
- `data/`: Settings, generation history, presets, image-editing data, and LoRA or merge working data
- `outputs/`: Generated images, edited images, and enhanced results
- `logs/`: Diagnostic logs for startup, ComfyUI, and the Asteria application

All of these locations are inside the extracted Asteria folder. Do not manually move `data/` or `outputs/` while Asteria is running.

### Eagle integration

Asteria can automatically send the final generated image to Eagle. Start Eagle, then configure its URL, destination folder, and **Automatically send completed generations to Eagle** under **External Tools** in Settings.

Intermediate files such as Hires.fix passes are not sent automatically. A failed Eagle transfer does not fail image generation. After restoring the connection, you can retry the transfer manually from the Library history.

### Discord generation

After saving the Bot connection and allowlists under Discord in Settings, publish the current configuration with **Publish to Discord** in Generate. Editing Generate later does not change the Discord configuration; publish again to apply changes. Discord generation is rejected while no profile is published.

The Bot token is never displayed after it is saved. Keep server, channel, and user allowlists as narrow as practical. Unpublish the profile and remove the token when the integration is no longer needed.

### Diagnostics, backup, and recovery

Diagnostics in Settings shows runtime readiness and actionable recovery. A diagnostics report is created only when you explicitly export it, and excludes prompts, tokens, absolute paths, process details, and raw logs. You can also clean completed records or safely restart a runtime whose Asteria ownership can be verified.

Maintenance in Settings shows compatibility information and creates backups. Restore requires both a preview and explicit confirmation. If an update or restore cannot complete, stop Asteria and run `Repair-Asteria.cmd`.

### Quick Q&A

#### Q. The browser does not open after I run `Start-Asteria.cmd`.

First check the final message in the CMD window. If first-run setup is still in progress, leave it open and wait for completion.

If an error is shown, check the NVIDIA driver, available disk space, internet connection, and security-software blocks. Then run `Repair-Asteria.cmd`. If required setup completed but application startup failed, also check the logs under `logs/launcher/`.

#### Q. The first download takes a long time or was interrupted.

Check the CMD window for a network or other error. If setup was interrupted, run `Start-Asteria.cmd` again. Verified files are reused and setup continues with the missing items.

#### Q. Asteria downloads something again on a later start.

Only a missing, corrupt, or manifest-changed component is reacquired. If the same item repeatedly fails, check free disk space and security software, then run `Repair-Asteria.cmd`.

#### Q. A model I added does not appear in the list.

Confirm that the file is in the folder for its model type, then refresh the list under **Model Assets** in Settings. If the model is linked, also confirm that its drive or network location is available.

#### Q. Asteria reports `A stale launcher record must be stopped or repaired before starting.`

Run `Stop-Asteria.cmd`, then run `Start-Asteria.cmd` again. If the message remains, run `Repair-Asteria.cmd`.

#### Q. Can I move the extracted folder?

Run `Stop-Asteria.cmd` and wait until Asteria is fully stopped, then move the entire folder. Do not move it while Asteria is running.

### Backup and uninstall

Before creating a backup, run `Stop-Asteria.cmd`. Save at least `data/`, `models/`, and `outputs/`. Include `logs/` if you want to preserve diagnostic information for troubleshooting.

When junctions or symbolic links are in use, backing up `models/` may not copy the actual files at their targets. Back up external model locations separately.

To uninstall Asteria, stop it with `Stop-Asteria.cmd`, then delete the extracted Asteria folder. This also deletes the settings, history, models, generated images, and logs stored inside that folder, so back up anything you need first.

### License and third-party components

The Asteria license is provided in `LICENSE`.

Provider information, pinned versions, and licenses for downloaded runtimes, packages, custom nodes, and models are recorded in `release-manifest.yaml`, `runtime-dependencies.yaml`, `model-dependencies.yaml`, and `notices/THIRD_PARTY_NOTICES.txt`. Third-party models, including starter packs, remain subject to the terms set by their providers.
