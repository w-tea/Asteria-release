# Krea 2／MOSS-Imageの画像制御

[日本語](#日本語) | [English](#english)

## 日本語

深度画像や色分け画像を使い、生成する画像の形や領域を指定できます。通常のプロンプトと、対応する絵柄LoRAも併用できます。

### 1. モデルを用意する

まずKrea 2 TurboまたはMOSS-Image SFTで通常生成できる状態にします。そのうえで、設定のモデル一覧から必要な制御モデルを追加します。

| 生成モデル | 追加する制御モデル | 使える入力 |
| --- | --- | --- |
| Krea 2 Turbo | Krea 2 Depth Control | 深度画像 |
| MOSS-Image SFT | MOSS-Image Color / Depth Control | 深度画像、色分け画像、両方の併用 |

手動で置く場合は、モデル保存場所の`loras`内へ入れてアセットを更新してください。制御モデルは通常のLoRA欄へ重ねて追加せず、次のControlNet欄で選びます。Krea 2の取得には利用条件の確認が必要です。モデルごとの利用条件は取得画面と配布元で確認してください。

### 2. 制御画像を選んで生成する

1. 生成画面でKrea 2またはMOSS-Imageを選び、プロンプトを入力します。
2. 「生成設定を開く」→「ControlNet」を選び、「追加」を押します。
3. 対応する「ControlNetモデル」と、入力の種類を選びます。Krea 2では深度画像、MOSS-Imageでは深度画像または色分け画像を選べます。
4. 制御画像をアップロードします。すでに用意した深度画像・色分け画像には、前処理の「用意した制御画像をそのまま使用」を選びます。
5. 強度、適用する開始・終了位置、リサイズ方法を確認し、生成します。入力と出力の縦横比が違う場合は、切り抜くか、余白を追加するかで構図も変わります。
6. 完成画像を開き、入力の形や領域がどう反映されたか確認します。生成履歴には制御画像と設定も残ります。

MOSS-Imageで両方を使う場合は、もう一つ追加して別の入力の種類を選びます。各ユニットには同じ制御モデルを指定してください。絵柄LoRAは通常のLoRA欄またはプロンプトから指定します。

### 入力と調整のポイント

- **深度画像**：近いものを白く、遠いものを暗くした画像です。写真を単に白黒へ変えた画像とは異なります。写真から作る場合は、導入済みの深度推定機能が利用できるときだけ、前処理で選べます。
- **色分け画像**：異なる領域を別の色で塗り分けた画像です。色は領域の識別に使われ、完成画像をその色に塗る指定ではありません。
- **強度と時間範囲**：制御画像をどの強さで、生成のどの区間へ渡すかを調整します。制御用LoRA自体の適用強度とは別です。MOSS-Imageの配布元の例では、開始0・終了0.5が使われています。
- **制御しない生成**：該当ユニットを無効にします。強度0、または開始と終了が同じ場合も、その入力は適用しません。
- **モデルが選べない場合**：制御モデルの導入後にアセットを更新してください。必要な実行環境が不足している場合は、設定から準備・修復します。

### 対応範囲

MOSS-Imageでは、幅・高さが16の倍数でなくても指定できます。内部の処理領域だけを16の倍数へ切り上げ、保存前に右端・下端の余剰部分を切り落とします。完成画像は指定サイズになり、履歴・metadataも同じ値を記録します。制御画像も同じ原点と処理領域へ合わせ、端数のために引き伸ばすことはありません。

この機能は構図や形の手掛かりを与えるものです。任意の姿勢、人物の所属、人体の正常性を保証しません。入力画像の形が出力にも強く残る場合があります。骨格ポーズ、輪郭専用の制御、Hires.fix、ADetailer、参照画像の部分編集を、新モデルで利用できるようにする機能ではありません。

## English

Use depth or color-region maps to guide shapes and regions while generating from a text prompt. Compatible style LoRAs can be used alongside the control model.

### 1. Prepare the models

First set up ordinary generation with Krea 2 Turbo or MOSS-Image SFT. Then install the optional control model from the model list in Settings:

| Generation model | Optional control model | Inputs |
| --- | --- | --- |
| Krea 2 Turbo | Krea 2 Depth Control | Depth maps |
| MOSS-Image SFT | MOSS-Image Color / Depth Control | Depth maps, color-region maps, or both |

For manual installation, place the file under `loras` in your model location and refresh assets. Select it in ControlNet, rather than also adding it to the regular LoRA stack. Krea 2 downloads require acceptance of its license. Check the download screen and publisher for each model's terms.

### 2. Select a map and generate

1. Select Krea 2 or MOSS-Image in Generate and enter your prompt.
2. Click **Open generation settings**, select **ControlNet**, then click **Add**.
3. Select the compatible **ControlNet model** and input type. Krea 2 uses depth; MOSS-Image also supports color-region maps.
4. Upload the control image. For a prepared depth or color-region map, select **Use the prepared map as is** under **ControlNet preprocessor**.
5. Review weight, guidance start/end and resize mode, then generate. Cropping and padding change the composition when input and output aspect ratios differ.
6. Inspect the finished image. History retains the control image and settings for reuse.

For MOSS-Image depth and color together, add a second unit with the other input type. Choose the same control model for every unit. Add a style LoRA through the regular LoRA controls or prompt.

### Input and adjustment tips

- **Depth map:** near objects are white and distant objects dark. A grayscale photograph is not a depth map. Photo-to-depth preprocessing is offered only when the installed depth-estimation component is available.
- **Color-region map:** different colors identify separate regions; they do not dictate the finished image's colors.
- **Weight and guidance interval:** these control the image signal and when it is supplied, separately from the control LoRA's own strength. The MOSS-Image publisher's example uses start 0 and end 0.5.
- **Without control:** disable the unit. A zero weight or equal start/end also leaves that input unapplied.
- **Missing model:** refresh assets after installation. Prepare or repair missing runtime components in Settings.

### Scope

MOSS-Image accepts widths and heights that are not multiples of 16. Only the internal canvas is rounded up; excess pixels at the right and bottom are cropped before saving. The final image, history and metadata retain the requested size. Control maps use the same origin and internal canvas, without stretching to fit the alignment padding.

These inputs guide composition and shape. They do not guarantee arbitrary poses, person ownership or correct anatomy. Input geometry may strongly influence the output. This does not add skeleton-pose control, edge-specific control, Hires.fix, ADetailer or masked reference-image editing to these models.

### Model references / モデル配布元

- [Krea 2 Depth Control](https://huggingface.co/Patil/Krea-2-depth-controlnet)
- [MOSS-Image Color / Depth Control](https://huggingface.co/darask0/MOSS-Image-color-depth-control)
