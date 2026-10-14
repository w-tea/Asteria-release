# 生成画像のチェック / Generated-image check

[日本語](#日本語) | [English](#english)

## 日本語

画像対応AIに、手足や身体の崩れを確認させる機能です。プロンプトどおりの人数、髪色、服装などを調べる機能とは別です。AIには見逃しや誤判定があるため、正常な身体を保証するものではありません。

### 設定する

1. 「設定」→「AIプロバイダー」を開きます。
2. 「生成画像のチェック」で「画像対応LLM」または「Codex」を選びます。LLMを使う場合は、保存済みの画像対応モデルの接続先を選びます。文章専用のモデルでは判定できません。Codexの利用可否は実行環境と対応モデルによって異なります。
3. 不合格画像を残したい場合は「不合格画像も保存」をONにします。
4. 次回から初期状態でチェックを有効にしたい場合は「生成画像のチェックを初期状態でONにする」を選び、「チェック設定を保存」を押します。生成画面でも、その生成に使うON／OFFを切り替えられます。

独自の人体判定器は現在選べません。接続するAIの判断能力によって結果が変わります。

### 保存前に試す

同じ欄にある「保存済み画像で試す」で、履歴の画像を選び「この画像で試す」を押します。「以前の画像を読み込む」で古い画像も選べます。任意のパスにある画像を、履歴へ登録せずに選ぶ欄ではありません。

この操作だけでは、設定を保存したり生成時のチェックをONにしたりしません。接続先の選択やモデル、待ち時間には画面の入力を使い、AIプロバイダーのURL等には保存済みの設定を使います。画像を変更・削除せず、生成履歴にも試験結果を追加しません。

- **崩れを見つけなかった場合**：AIが明確な崩れを見つけなかった、という結果です。完全に正常という保証ではありません。
- **崩れを指摘した場合**：説明が実際の画像と合っているか確認してください。
- **判定できなかった場合**：接続先や対応モデル、待ち時間を確認してください。接続失敗を画像の崩れとは扱いません。

判定中は「判定を中止」で止められます。画像やチェック設定を変更すると、それまでの結果は消え、進行中の判定も中止します。

### 生成時の動き

チェックをONにした生成では、Hiresや部分補正の前に画像を確認します。不合格なら、その画像にHiresや部分補正をかけません。「不合格画像も保存」がOFFの場合は画像を保存せず、履歴には判定を残します。ONの場合は不合格の画像も残します。

単発生成はその結果で終了します。自動リピートは不合格でも次の生成へ進み、判定不能や接続エラーでは停止します。小さな手指は拡大・補正前では判別しづらいことがあります。Hires後に再判定する機能ではありません。

### 画像の送信

選んだAI接続先には判定用の画像を送ります。プロンプトや画像内のメタデータは送りません。外部サービスを選んだ場合、画像はPCの外へ送られます。接続先を確認してから実行してください。一枚の試験だけでAIの精度は判断できません。

## English

This feature asks an image-capable AI to check for anatomical defects. It is separate from checking whether an image matches requested people, hair, clothing, or other prompt conditions. The AI may miss defects or report false positives; it does not guarantee correct anatomy.

### Configure the check

1. Open Settings → AI providers.
2. Under Generated image check, select Image-capable LLM or Codex. For an LLM, choose a saved connection with an image-capable model. Text-only models cannot judge the image. Codex availability depends on the installed environment and supported models.
3. Enable Also save rejected images if you want to keep rejected results.
4. To enable checking by default, select Enable checking by default and click Save check settings. The Generate screen also has a per-generation ON/OFF control.

An Asteria-native anatomy classifier is not currently available. Results depend on the selected AI’s capabilities.

### Try before saving

In Try with a saved image, select an image from generation history and click Check this image. Use Load earlier images to browse older results. This selector does not accept arbitrary files that are not registered in generation history.

Trying a check does not save settings or enable checking for future generations. The provider selection, model, and timeout come from the current check form, while the AI provider’s URL and other connection details come from its saved definition. The trial does not modify or delete the image and does not add a result to generation history.

- **No defect found** means the AI did not identify a clear defect, not that the image is guaranteed normal.
- **Defect reported** means you should compare its explanation with the actual image.
- **Inconclusive** means you should check the connection, model support, or timeout. A connection failure is not an anatomical defect.

Cancel check stops an ongoing trial. Changing the image or check settings clears the previous result and cancels an ongoing request.

### During generation

Enabled checks run before Hires or regional enhancement. Rejected images skip those stages. With Also save rejected images disabled, the image is not saved, but the verdict remains in history. With it enabled, the rejected image is kept.

A single generation ends with that result. Auto-repeat continues after a rejection, but stops if the check is inconclusive or the connection fails. Small fingers may be hard to assess before upscaling or enhancement. This feature does not run a second check after Hires.

### Image transmission

The selected AI connection receives an image for evaluation, without the prompt or embedded image metadata. An external service receives the image outside your PC. Check the destination before running the trial. One image does not establish the AI’s accuracy.
