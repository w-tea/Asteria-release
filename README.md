# Asteria V1.0.3 Windows Portable

[日本語](#日本語) | [English](#english)

## 日本語

### 概要

Asteriaは、ローカル環境で画像生成、編集、比較、整理、LoRA学習、モデルマージを行うWindows向けAI画像制作スタジオです。ZIPを展開して起動すると、機能に必要なPython、Node.js、ComfyUI、パッケージ、カスタムノード、機能用データを初回セットアップで取得します。画像生成モデルは同梱していません。

### インストール方法

1. 配布ZIPを、十分な空き容量があり書き込み可能なフォルダーへ完全に展開します。
2. 展開したフォルダー内の`Start-Asteria.cmd`をダブルクリックします。
3. 初回はCMDのセットアップ画面で、起動に必要なPython、Node.js、ComfyUI、パッケージ、カスタムノード、機能用データを自動取得します。CMD上でスターターパックは選択しません。取得と検証が完了するまでウィンドウを閉じずに待ちます。
4. 必須セットアップが完了すると、Asteriaがブラウザーで開きます。
5. SDXL／Animaスターターパックが必要な場合は、AsteriaのSettingsにある「起動準備」から選択して取得します。片方だけ、両方、または両方をスキップできます。

Windows側へ事前に必要なのは対応するNVIDIAドライバーです。CUDA Toolkit、Python、Node.js、ComfyUIを個別にインストールする必要はありません。セットアップ中にウィンドウを閉じても、検証済みの取得物は次回の続きで再利用されます。

### 必要最低限のスペックと動作環境

- 64-bit版Windows 10またはWindows 11
- CUDA対応のNVIDIA GPU（AMD／Intel GPU、CPUのみの実行は未対応）
- NVIDIAドライバー 570.65以上
- x64 CPU、インターネット接続、展開先への書き込み権限
- 初回取得、選択したモデル、生成物を保存できる空き容量（取得計画の正確な必要量はセットアップ画面に表示されます。機能用データだけでも数十GBになる場合があります）

macOSとLinuxはV1.0.3では未対応です。必要なGPUメモリとシステムメモリは、使用するモデル、解像度、LoRA学習などの処理によって変わります。高解像度生成や学習には、より多くのGPUメモリとシステムメモリが必要です。

### 起動方法

- 起動: `Start-Asteria.cmd`
- 終了: `Stop-Asteria.cmd`
- 取得失敗や破損の修復: Asteriaを終了してから`Repair-Asteria.cmd`

2回目以降は取得済みコンポーネントを検証し、変更がなければダウンロードをスキップします。SDXLとAnimaのスターターパックは初期状態で未選択で、CMDではなくAsteria起動後のSettings「起動準備」から取得します。Animaの取得には画面に表示される非商用ライセンスへの同意が必要です。

自分のモデルは`models/`配下へ配置できます。Portable版V1.0.3ではSettingsの外部モデルルート設定は使用せず、別の場所にある既存モデルを共有するときはWindowsのジャンクションまたはシンボリックリンクを使用します。複数の外部モデルルートをSettingsから登録する正式対応はV1.1で検討します。

主な保存場所:

- モデル: `models/`
- 設定、履歴、作業データ: `data/`
- 生成画像: `outputs/`
- 診断ログ: `logs/`

### よくある質問（簡易Q&A）

#### Q. `Start-Asteria.cmd`を実行してもブラウザーが開きません。

セットアップ用ウィンドウの表示を確認し、処理が完了するまで待ってください。失敗した場合は、NVIDIAドライバー、空き容量、インターネット接続、セキュリティソフトによる遮断を確認してから`Repair-Asteria.cmd`を実行します。

#### Q. 初回ダウンロードが長い、または途中で止まりました。

初回は数十GBを取得する場合があります。Asteriaを終了して再度`Start-Asteria.cmd`を実行すると、検証済みファイルを再利用し、不足分から続行します。

#### Q. 2回目以降もダウンロードされます。

不足、破損、または配布manifestの変更が検出されたコンポーネントだけを再取得します。毎回同じ項目で止まる場合は、空き容量とセキュリティソフトを確認して`Repair-Asteria.cmd`を実行してください。

#### Q. 自分のモデルはどこへ置きますか。

`models/`配下へ配置します。SDXL／Animaスターターパックを取得しなくても、自分の対応モデルを利用できます。

`models/checkpoints/`、`models/diffusion_models/`、`models/vae/`、`models/text_encoders/`、`models/loras/`、`models/embeddings/`、`models/controlnet/`、`models/bbox/`、`models/upscalers/`は初回起動時に自動作成されます。モデルの種類に合うフォルダーへ配置してください。`sdxl`や`anima`など、さらに下のサブフォルダー分けは任意です。

#### Q. 生成完了時に画像をEagleへ自動送信できますか。

できます。Eagleを起動し、SettingsのExternal ToolsでEagleのURL、送信先フォルダー、「生成完了時にEagleへ自動送信」を設定して保存してください。自動送信するのは最終生成画像だけで、Hires.fixの中間画像などは送信しません。Eagleへの送信に失敗しても画像生成自体は完了し、履歴から手動で再送できます。

#### Q. 別ドライブの既存モデルをコピーせずに共有できますか。

できます。まず`Stop-Asteria.cmd`でAsteriaを完全に終了し、モデル種別フォルダー内またはその下の妥当な位置へ、任意の名前でディレクトリリンクを作成します。一般利用者には管理者権限を要求しにくいジャンクションを推奨します。

```bat
mklink /J "<Asteriaのモデル種別フォルダー内の任意パス>" "<既存モデルフォルダー>"
```

たとえば`models\checkpoints\my-library`という名前は一例で、`my-library`は任意に変更できます。ディレクトリのシンボリックリンクを使う場合はWindowsのDeveloper Modeまたは管理者権限が必要になることがあります。ファイル単位のシンボリックリンクにも対応しています。

リンク作成後は`Start-Asteria.cmd`で再起動し、SettingsのModel Assetsで状態を更新してください。リンク先が利用不能な場合、そのモデルは利用不可として扱われます。Asteriaはリンク先のモデルを移動、削除、修復せず、`Repair-Asteria.cmd`もリンク先を変更しません。`models/checkpoints`や`models/vae`など管理モデルの取得先そのものをリンクにすると、読み取りはできても安全のためスターターモデル取得やRepairが拒否される場合があります。

#### Q. 展開したフォルダーを移動できますか。

Asteriaを`Stop-Asteria.cmd`で完全に終了してから、フォルダー全体を移動してください。起動中には移動しないでください。

#### Q. バックアップやアンインストール方法は。

バックアップでは、少なくとも`data/`、`models/`、`outputs/`を保存してください。アンインストールは展開フォルダーの削除で完了しますが、フォルダー内の設定、履歴、モデル、生成画像も削除されます。

第三者コンポーネントの提供元、固定バージョン、ライセンス情報は`release-manifest.yaml`、`runtime-dependencies.yaml`、`model-dependencies.yaml`および`notices/`に収録されています。Asteria本体のライセンスは`LICENSE`です。

---

## English

### Overview

Asteria is a Windows AI image production studio for local image generation, editing, comparison, organization, LoRA training, and model merging. After you extract and start the Portable package, first-run setup downloads the required Python, Node.js, ComfyUI, packages, custom nodes, and functional assets. Generation model weights are not bundled.

### Installation

1. Fully extract the distribution ZIP to a writable folder with ample free space.
2. Double-click `Start-Asteria.cmd` in the extracted folder.
3. On the first run, the CMD setup window automatically downloads the Python, Node.js, ComfyUI, packages, custom nodes, and functional assets required to start Asteria. Starter packs are not selected in CMD. Keep the window open until acquisition and verification finish.
4. Asteria opens in the browser after the required setup completes.
5. If you want an SDXL or Anima starter pack, select it from **Settings > Ready to start** after Asteria opens. You can install either pack, both packs, or skip both.

Only a compatible NVIDIA driver must be installed on Windows beforehand. You do not need to install the CUDA Toolkit, Python, Node.js, or ComfyUI separately. If setup is interrupted, verified completed downloads are reused on the next run.

### Minimum requirements and supported environment

- 64-bit Windows 10 or Windows 11
- A CUDA-capable NVIDIA GPU; AMD/Intel GPUs and CPU-only operation are unsupported
- NVIDIA driver 570.65 or newer
- An x64 CPU, internet access, and write access to the extracted folder
- Enough free space for first-run acquisitions, selected models, and outputs; setup displays the exact planned download size, and functional assets alone may require tens of GB

macOS and Linux are unsupported in V1.0.3. GPU-memory and system-memory needs depend on the selected model, resolution, and workload such as LoRA training. High-resolution generation and training require more memory.

### Starting Asteria

- Start: `Start-Asteria.cmd`
- Stop: `Stop-Asteria.cmd`
- Retry or repair an incomplete or corrupt managed component: stop Asteria, then run `Repair-Asteria.cmd`

Later starts verify acquired components and skip downloads when nothing changed. The SDXL and Anima starter packs are optional, initially unchecked, and installed from **Settings > Ready to start** after Asteria opens rather than from CMD. Downloading Anima requires acceptance of the displayed non-commercial license.

Place your own models under `models/`. Portable V1.0.3 does not use the external model-root setting in Settings. To share an existing library elsewhere, use a Windows junction or symbolic link. Formal Settings support for multiple external model roots is being considered for V1.1.

Main storage locations:

- Models: `models/`
- Settings, history, and working data: `data/`
- Generated images: `outputs/`
- Diagnostic logs: `logs/`

### Quick Q&A

#### Q. The browser does not open after I run `Start-Asteria.cmd`.

Check the setup window and wait for it to finish. If setup fails, check the NVIDIA driver, free disk space, internet access, and security-software blocks, then run `Repair-Asteria.cmd`.

#### Q. The first download is slow or was interrupted.

The first run may download tens of GB. Stop Asteria and run `Start-Asteria.cmd` again; verified files are reused and acquisition continues with missing items.

#### Q. Asteria downloads something again on a later start.

Only a missing, corrupt, or manifest-changed component is reacquired. If the same item repeatedly fails, check free space and security software, then run `Repair-Asteria.cmd`.

#### Q. Where should I put my own models?

Place them under `models/`. You can skip the SDXL and Anima starter packs and use your own compatible models.

`models/checkpoints/`, `models/diffusion_models/`, `models/vae/`, `models/text_encoders/`, `models/loras/`, `models/embeddings/`, `models/controlnet/`, `models/bbox/`, and `models/upscalers/` are created automatically on first launch. Place each model in the folder for its type. Additional subfolders such as `sdxl` or `anima` are optional.

#### Q. Can Asteria send completed images to Eagle automatically?

Yes. Start Eagle, then configure its URL, destination folder, and **Automatically send completed generations to Eagle** under **Settings > External Tools**. Only final generation outputs are sent; intermediate Hires.fix images are excluded. If Eagle delivery fails, generation still completes and you can retry manually from history.

#### Q. Can I share an existing model library on another drive without copying it?

Yes. First stop Asteria completely with `Stop-Asteria.cmd`. Create a directory link, with any name, at a suitable location inside or below the corresponding model-type folder. A junction is recommended for most Windows users because a directory symbolic link may require Developer Mode or administrator privileges.

```bat
mklink /J "<any path inside an Asteria model-type folder>" "<existing model folder>"
```

For example, `models\checkpoints\my-library` is only an example; `my-library` is not a required name. File symbolic links are also supported. After creating the link, start Asteria again and refresh Model Assets in Settings.

If the target becomes unavailable, the linked model becomes unavailable. Asteria does not move, delete, or repair linked models, and `Repair-Asteria.cmd` does not modify link targets. If a managed destination itself—such as `models/checkpoints` or `models/vae`—is a link, reading may work, but starter-model acquisition or Repair can be refused for safety.

#### Q. Can I move the extracted folder?

Run `Stop-Asteria.cmd` and wait until Asteria is fully stopped, then move the entire folder. Do not move it while Asteria is running.

#### Q. How do I back up or uninstall Asteria?

For a backup, save at least `data/`, `models/`, and `outputs/`. To uninstall, delete the extracted folder; this also deletes settings, history, models, and outputs stored inside it.

Provider, pinned version, and third-party license information is included in `release-manifest.yaml`, `runtime-dependencies.yaml`, `model-dependencies.yaml`, and `notices/`. The Asteria license is in `LICENSE`.
