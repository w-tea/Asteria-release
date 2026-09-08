# 項目別プロンプトの配置

項目別入力では、ポジティブプロンプトを「ベース」「キャラクター」「詳細」などに分けて管理できます。さらに、必要な項目だけを画像内の配置対象にして、対象ごとの範囲と条件を設定できます。

## dev版の制限（2026年9月8日）

配置はまだ開発中です。処理の重さを抑えるため、追加の反復生成をやめ、最初の生成の前半だけ位置を誘導する方式へ変更しました。人数や位置が指定どおりにならない、髪色・髪型・服装が別の対象へ移る、といった問題は残っています。配置の品質が完成した版ではありません。

Hires.fixとRefinerでは配置の誘導を追加せず、通常の拡大・再描画を使います。配置を無効にすれば通常の生成を使えます。配置した範囲は、描画を切り抜く枠ではなく、対象を寄せたい位置と大きさの目安です。

## 基本の使い方

配置を設定しない項目は、これまでどおり画像全体へ反映します。配置を使わない生成では、配置用の追加処理は行いません。

配置したい項目で「配置を設定」を押すと、専用画面が開きます。選んだ項目の番号、名前、入力内容を引き継ぎます。文章の内容から人物や物体を自動で分けることはありません。内容を確認し、必要なら別の項目にも配置を設定するか、配置画面で対象を追加してください。

対象数に固定の上限はありません。対象を増やすほど処理時間とメモリ使用量が増えるため、4件目以降は注意が表示されます。注意が出ても、利用者が実行を選べば生成できます。

## 範囲と形

対象ごとに楕円または長方形を選べます。範囲は中央のキャンバスで移動・拡大縮小し、配置の強さと境界のなじみを自動または手動で調整します。設定値は画像の基準キャンバスに対する相対値として保存されるため、画面上のピクセル座標そのものは保存しません。

旧Animaのデータに自由な形の範囲指定が含まれる場合は、旧形式の範囲として読み込みます。移動と拡大縮小はできますが、楕円や長方形へ変更するまで元の輪郭を保ちます。旧設定を読み込んだ場合も、通常のプロンプトは失われません。

配置の強さを上げると指定範囲へ結び付きやすくなりますが、構図や画質へ影響する場合があります。境界のなじみを上げると周囲へ滑らかにつながります。どちらも結果を保証する値ではありません。

Asteriaは、対象ごとの文章と範囲をモデル内部のAttentionで関連付けます。ControlNetやLLLiteの制御画像を、そのまま配置機能の代わりに使う方式ではありません。ControlNetやLLLiteを併用した場合は、制御画像が位置や形を補い、Asteriaの配置処理が文章と対象の対応付けを補います。

対象が重なる場合は、前後関係を自動または手動で選べます。これは生成中に特徴をどちらへ寄せるかの手掛かりであり、後処理で画像を重ねる機能でも、完全な遮蔽順を保証する機能でもありません。

## 背景画像、Hires、モデル系統

img2imgやinpaintでは、入力画像を背景としてキャンバスに表示し、その上で配置を設定できます。Asteriaは、生成に使うcheckpointの構造を内部で確認します。SDXL／Illustrious系では共通の24ブロック構造を使い、Animaでは28／40／52ブロックの違いを判別します。利用者がモデル系統を選ぶ必要はありません。構造を確認できないモデルでは配置を付けた生成を始めず、別のモデルへ切り替えるよう案内します。

現在のdev版では、Hires.fixは最初に生成した画像を通常の処理で拡大・再描画します。Hires用に別のcheckpointを選んだ場合も、配置の誘導は追加しません。Topaz Photo AIが担当するのは画像の拡大です。拡大後の再描画で構図や属性が変わる場合があります。

Anima 2.9B／3.8Bでは、LLLiteが生成結果に作用することを確認しています。ただし、現在の位置や構図の制御は実験段階です。狙った位置や構図を保証するものではないため、強さと範囲を調整しながら結果を確認してください。

## 画像サイズを変えたとき

生成サイズを大きくした場合、配置の中心と範囲を保ったまま基準キャンバスを更新します。小さくした場合や縦横比を変えた場合、範囲がキャンバスの外へ出る可能性があるため、実行前に確認が表示されます。画面の案内に従って範囲を調整するか、配置を維持するか、変更を中止してください。

## 失敗したとき

範囲の重なりが成立しないinpaintでは、対象の位置を調整する、今回だけ対象を外す、配置を無効にして生成する、操作を中止する、のいずれかを選べます。モデルや依存機能が不足している場合は、画面の案内に従って設定を直すか、配置なしで生成してください。配置を無効にしても、保存した対象と範囲は削除されません。

配置は、対象が指定位置に現れやすくなるよう生成を誘導する機能です。モデル、Seed、プロンプト、LoRA、ControlNet、画像サイズなどによって結果は変わり、位置、人数、遮蔽、輪郭を保証するものではありません。NovelAIと同じ精度や挙動を意味する機能ではありません。

## 過去データの移行

旧Animaの領域別プロンプトを含む画像や履歴は、読み込み時に移行できる場合があります。元の色分け画像が必要なデータでは、画像を選び直してから配置を確認してください。移行できない場合でも、通常のプロンプトは再利用できます。保存後の新規生成は、Asteriaの配置形式で処理します。

---

## English

Sectioned prompt input can keep the positive prompt in separate sections such as Base, Character, and Details. You can also place selected sections within the image and configure a separate area for each target.

### Dev limitations (September 8, 2026)

Placement remains experimental. To reduce overhead, additional iterative generation passes have been removed; placement now guides only the early part of base sampling. Incorrect counts or positions and mixed hair, hairstyle, and clothing assignments remain unresolved. This is not a completed placement-quality release.

Hires.fix and Refiner use normal enlargement/refinement without additional placement guidance. Disable placement to use regular generation. Each area expresses a preferred position and size, not a hard drawing boundary or cutout.

### Basic workflow

Sections without placement continue to affect the whole image. A generation without placement does not use the additional spatial processing path.

Select **Set placement** for a section to open the placement editor. It inherits the selected section's number, name, and prompt. Asteria does not automatically split people or objects from the text. Review the content, then place another section or add more targets in the editor as needed.

There is no fixed target limit. More targets require additional processing time and memory, so Asteria shows an advisory from the fourth target onward. The advisory does not disable generation.

### Area and shape

Each target can use an ellipse or rectangle. Move and resize the area on the canvas, then leave strength, edge blending, and front-to-back priority on Auto or adjust them manually. Asteria stores the placement relative to the reference canvas instead of saving screen pixel coordinates.

When imported legacy Anima data contains a free-form area, Asteria keeps that outline until you replace it with an ellipse or rectangle. Importing legacy placement does not discard the regular prompt.

Strength influences how closely a section is associated with its target area, but higher values can affect composition and image quality. Edge blending controls how smoothly the target connects to its surroundings. Neither value guarantees a particular result.

Asteria associates each target's text with its area inside the model's attention layers. It does not use a ControlNet or LLLite control image as a substitute for prompt placement. When combined, ControlNet or LLLite can guide position and shape while Asteria's placement path reinforces which text belongs to each target.

Overlapping targets can use automatic or manual front-to-back priority. This priority guides conditioning during generation. It is not post-processing compositing and does not guarantee complete occlusion.

### Source images, Hires.fix, and model families

For img2img and inpaint, the source image appears behind the placement canvas. Asteria inspects the selected checkpoint structure internally. SDXL and Illustrious use the shared 24-block path, while Anima is classified by its 28-, 40-, or 52-block structure. You do not need to select a model family. If Asteria cannot verify a supported structure, it stops before generation and asks you to choose another model.

In the current dev build, Hires.fix enlarges and refines the base image through the normal path. No additional placement guidance runs, including when a separate Hires checkpoint is selected. Topaz Photo AI performs enlargement only. Subsequent refinement may change the composition or attributes.

LLLite has a confirmed effect on outputs from Anima 2.9B and 3.8B. Its current position and composition control remains experimental and does not guarantee the intended layout. Adjust the strength and area while reviewing the results.

### Changing image size

When you enlarge the generation size, Asteria keeps the relative center and extent of each target. Reducing the size or changing the aspect ratio may move part of a target outside the canvas. Asteria then asks whether to adjust the placement, keep it as-is, or cancel the size change.

### Recovery

If an inpaint target does not overlap the editable area, you can adjust placement, exclude that target for the current run, disable placement, or cancel. If a selected model or required component is unavailable, follow the displayed recovery guidance or generate without placement. Disabling placement does not delete the saved targets.

Spatial prompt placement guides the model toward the selected areas. Results still depend on the model, Seed, prompt, LoRAs, ControlNet, and image size. It does not guarantee position, subject count, occlusion, or outlines, and it does not claim NovelAI-equivalent behavior.

### Importing older data

Asteria can migrate some images and history entries that contain the earlier Anima regional-prompt format. If the original color map is required, select it again and review the imported placement. Even when placement cannot be migrated, the regular prompt remains available. New generations are saved with Asteria's placement format.
