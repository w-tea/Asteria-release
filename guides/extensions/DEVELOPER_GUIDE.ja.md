# Asteria 拡張機能 開発者ガイド

[English](DEVELOPER_GUIDE.md) | [日本語](DEVELOPER_GUIDE.ja.md)

このガイドでは、Asteria V1.2向けの拡張機能を作成し、検証して配布するまでの手順を説明します。

## 実行方式を選ぶ

Asteriaには、権限の異なる2種類の拡張機能があります。

- **標準拡張**: 分離されたブラウザー画面で動き、Asteriaが許可した操作だけを仲介機能へ依頼します。通常はこちらを選んでください。
- **フルアクセス拡張**: 作者が用意したPythonやJavaScriptをAsteria本体と同じ権限で動かします。Asteriaの公開APIだけでは作れない機能に使用します。

### 標準拡張でできること

標準拡張は、`sandbox="allow-scripts"`を指定したiframe内で動きます。本体画面のDOM、Cookie、`localStorage`、Piniaストアには直接触れられません。V1.2で利用できる範囲は次のとおりです。

- 左メニューに専用画面を追加する
- Generate、Image Edit、Media Library、LoRA Training、Merge Studio、Settingsの決められた場所へパネルや操作を追加する
- Asteriaの既存画面へ移動する
- 申告した`prompt.read`権限で、Generateに入力中のプロンプトとネガティブプロンプトを読み込む
- マニフェストに記載した通信先へ、Asteriaの仲介機能を通して接続する
- iframe内でJavaScriptまたはWASMを実行する

たとえば、入力中のプロンプトを点検する画面、申告済みの外部APIへプロンプトを送り提案を表示する補助ツール、Asteriaの画面を順番に案内する作業ナビゲーションは標準拡張で作れます。

### 標準拡張ではできないこと

次の処理は標準拡張から直接実行できません。

- ComfyUIのカスタムノードやPythonパッケージを自動で導入する
- `pip install`、`npm install`、`setup.py`、独自のインストールスクリプトを実行する
- Pythonバックエンド、Node.js、任意のローカル処理を実行する
- 任意のEXE、PowerShell、コマンドプロンプトを起動する
- Asteriaの内部モジュール、データベース、非公開ストア、非公開DOMを直接操作する
- PC内のファイルやフォルダーを読み書きする
- Media Libraryの選択項目、タグ、評価、メモを読み書きする
- 生成を開始する、成果物を登録する、画像や動画を一括変換する
- マニフェストに記載していないURLへ接続する
- ローカルネットワークや`localhost`を走査する
- 独自のCUDA、PyTorch、ネイティブGPUコードを直接実行する
- Asteriaの終了後もプロセスや常駐処理を残す

これらが必要な場合はフルアクセス拡張を選びます。標準拡張の制限を回避するために、外部プログラムやローカルサーバーを隠して起動してはいけません。

### フルアクセス拡張でできること

フルアクセス拡張は、現在のコード実行型拡張と同じ自由度を持ちます。独自のComfyUIノード、ネイティブライブラリ、CUDA処理、外部プログラム、独自サーバー、Asteriaの公開APIでは表せない処理を組み込めます。

一方で、Asteria本体と同じ利用者権限で動くため、PC内のファイル、ネットワーク、プロセスへアクセスできます。Asteriaは導入前にこの点を明示し、利用者が承認した場合だけ有効にします。Safe Mode、無効化、元に戻す機能は復旧手段であり、悪意のあるコードによる情報流出や破壊を事後に取り消す機能ではありません。

### どちらを選ぶか

| 作りたい機能 | 選ぶ方式 |
| --- | --- |
| 入力中のプロンプトを読み、提案を表示する | 標準拡張 |
| 申告したHTTPS APIまたはlocalhostへ仲介通信する | 標準拡張 |
| Asteriaの既存画面を案内する専用画面やパネル | 標準拡張 |
| Media Libraryのファイル処理、タグ更新、生成の開始 | フルアクセス拡張 |
| Python、Node.js、外部ライブラリを使う処理 | フルアクセス拡張 |
| 独自ComfyUIノード、CUDA処理、ネイティブライブラリ | フルアクセス拡張 |
| 任意の外部プログラムやローカルサーバー | フルアクセス拡張 |
| Asteriaの内部実装を直接変更する処理 | 拡張機能ではなくAsteria本体への提案を検討 |

迷う場合は標準拡張から始めてください。公開APIで足りない場合は、必要な操作をAsteriaのAPIとして追加できないか先に検討します。

## 権限と責任

標準拡張は、使うAsteria機能と通信先をマニフェストに記載します。V1.2で指定できるAsteria権限は`prompt.read`だけです。外部通信は完全なHTTPSオリジンを、localhost通信はホスト名とポートを個別に記載します。HTTPSオリジンのホスト名は小文字にし、既定のポート`443`は省略してください。Asteriaは、導入時に表示した範囲だけを仲介します。

フルアクセス拡張では、`network`（ネットワーク）、`filesystem`（ファイル）、`subprocess`（外部プロセス）のうち、使う機能を正確に記載してください。この記載は利用者が判断するための情報であり、フルアクセス拡張のOS権限を制限する仕組みではありません。ソース、ライセンス、取得元を確認してから配布してください。

フルアクセス拡張はAsteriaプロセス内で読み込まれます。ただし、コアの非公開DOM、Piniaストア、データベース、ランタイムのノードIDは安定した公開契約ではありません。Asteriaが提供する登録コンテキストとAPIを優先してください。

## ひな形を作る

Extension Kitの依存関係を開発環境で一度用意し、ひな形を作成します。

```powershell
cd extension-kit
npm install
cd ..
node extension-kit/scripts/create-extension.mjs --output extensions/my-extension
```

このコマンドは標準拡張を作ります。フルアクセス拡張を作る場合は、末尾に`--execution-level full_access`を追加します。ID、表示名、セマンティックバージョンを指定しない場合は対話形式で入力します。IDは`com.example.my-extension`のような逆ドメイン形式にしてください。

生成されたライセンスの `placeholder`（仮置き値）は、ビルド前に実際のライセンスID、URL、本文へ置き換えます。placeholderを残したパッケージは拒否されます。

## Manifest

マニフェストは `asteria-extension.json` に記述し、`schemas/asteria-extension-v1.schema.json` に従います。最上位で使うフィールドは次のとおりです。

```json
{
  "schema_version": "asteria.extension.v1",
  "id": "com.example.my-extension",
  "name": "My Extension",
  "version": "1.0.0",
  "asteria": { "minimum": "1.2.0", "maximum_exclusive": "2.0.0" },
  "entrypoints": {
    "backend": "backend.extension:create_extension",
    "frontend": "frontend/dist/extension.mjs",
    "frontend_element": "asteria-com-example-my-extension"
  },
  "contributions": {
    "workspaces": [{ "id": "workspace.main", "name": "Main workspace", "surface": "independent_workspace" }],
    "panels": [{ "id": "panel.generate", "name": "Generate panel", "surface": "generate.inspector_panel" }],
    "actions": [{ "id": "action.generate", "name": "Generate action", "surface": "generate.inspector_action" }]
  },
  "execution_level": "standard",
  "permissions": {
    "asteria": ["prompt.read"],
    "network_hosts": ["https://api.example.com"],
    "localhost": [{ "host": "127.0.0.1", "port": 12345 }]
  },
  "license": { "id": "MIT", "url": "https://opensource.org/license/mit" }
}
```

`schema_version`、`id`、`name`、`version`、`asteria`、`entrypoints`、`contributions`、`license`が必須です。`execution_level`を省略した従来のパッケージは、フルアクセス拡張として扱います。標準拡張では`execution_level`を`standard`にして、必要な`permissions`だけを記載してください。標準拡張に`capabilities`、`backend.api_router`、`backend.operation`は指定できません。

フルアクセス拡張では`execution_level`を`full_access`にします。`permissions`は空にし、`network`、`filesystem`、`subprocess`のうち実際に使うものを`capabilities`へ記載します。`schema_version`は`asteria.extension.v1`、`id`は一意な識別子、`name`は表示名、`version`はセマンティックバージョンです。`asteria`は`minimum`以上、`maximum_exclusive`未満の対応範囲を表します。

Asteriaへ登録する画面・操作の単位を「提供項目」と呼びます。各提供項目の`id`は、一つの拡張機能内で一意にしてください。マニフェストに記載した提供項目はすべて登録します。記載のない項目や重複した項目を登録すると、読み込みに失敗します。

利用できる`surface`は`independent_workspace`、`generate.inspector_panel`、`generate.inspector_action`、`image_edit.inspector_panel`、`image_edit.inspector_action`、`media_library.details`、`media_library.context_action`、`lora_training.tool`、`lora_training.action`、`merge_studio.tool`、`merge_studio.action`、`settings.integration_card`、`backend.api_router`、`backend.operation`です。末尾の2つはフルアクセス拡張だけで利用できます。

## 標準拡張の画面と仲介機能

標準拡張のフロントエンドは、`asteria.extension.standard.frontend.v1`契約の`asteriaStandardExtension`を公開します。`mount`へ渡される`root`の中だけへ表示し、Asteriaへの要求は`bridge.request`を使います。

```ts
export const asteriaStandardExtension = {
  contractVersion: 'asteria.extension.standard.frontend.v1',
  async mount({ root, bridge }) {
    const { prompt, negativePrompt } = await bridge.request('prompt.read', null)
    const output = document.createElement('pre')
    output.textContent = `${prompt}\n\n${negativePrompt}`
    root.replaceChildren(output)
  },
}
```

V1.2の仲介機能は、`navigate`、`command`、`prompt.read`、`network.fetch`に対応します。`prompt.read`にはマニフェストの同名権限が必要です。`network.fetch`は`GET`と`POST`だけを受け付け、マニフェストに記載したHTTPSオリジンまたはlocalhostのホスト名・ポートと完全に一致する場合だけ通信します。リダイレクト、資格情報を含むURL、未申告の接続先、外部ホストからプライベートIPへの解決は拒否されます。同時に処理できる通信は4件までです。上限を超えた場合は、処理中の通信が終わってから再試行してください。

Asteria本体とComfyUIが使う3000、3001、5173、8000、8188番ポートは、標準拡張の接続先に指定できません。標準拡張から既存のローカルサービスへ接続する場合は、これらと重ならないポートを使ってください。Asteriaの内部APIやComfyUIを直接操作する必要がある機能は、フルアクセス拡張として実装します。

標準拡張に含まれる`backend/extension.py`は、V1パッケージの形を保つためのファイルです。Asteriaは読み込みも実行もしません。Python処理が必要ならフルアクセス拡張を選んでください。

## API RouterとOperation Executor（フルアクセス拡張）

バックエンドのエントリーポイントは`module.path:function`形式です。たとえば`backend.extension:create_extension`は公開登録コンテキストを受け取り、APIルーター（HTTPルートをまとめて登録する単位）や提供項目を登録します。

```python
from fastapi import APIRouter
from app.extensions.public_api import OperationExecutionContext, OperationExecutionResult


class MyExecutor:
    version = "1"

    def resource_claims(self, request_snapshot: dict) -> tuple[str, ...]:
        return ("workspace:my-extension",)

    def execute(self, context: OperationExecutionContext) -> OperationExecutionResult:
        context.report_progress(1, 1, "item", phase="completed")
        return OperationExecutionResult()

def create_extension(context) -> None:
    router = APIRouter()
    @router.get("/status")
    def status() -> dict[str, str]:
        return {"status": "ready"}
    context.register_router("router.api", router)
    context.register_workspace("workspace.main")
    context.register_panel("panel.generate")
    context.register_action("action.generate")
    context.register_operation("operation.run", MyExecutor())
```

AsteriaはAPIルーターを `/api/extensions/<extension-id>/` 以下へ配置します。拡張機能は相対パスだけを宣言し、コアのデータベースへ直接アクセスしません。

時間のかかる処理をAsteriaへ渡す実行オブジェクトは、公開プロトコルの`version`、変更されないリクエスト情報から算出する`resource_claims`、実行時に呼び出す`execute`を提供します。`context.register_operation(contribution_id, executor)`で`backend.operation`の提供項目として登録してください。Asteria内部の実行管理を直接操作しないでください。

V1.2では、実行オブジェクトの`execute`はAsteriaが提供する`OperationExecutionContext`を受け取り、`OperationExecutionResult`を返します。任意の`dict`を戻り値にしないでください。`extension-kit/src/backend/asteria_extension.py`は、登録コンテキストと公開プロトコルを開発時に確認するための型定義です。実行時の正式な型は`app.extensions.public_api`から読み込み、内部サービスモジュールを直接参照しません。

`app.extensions.public_api`は、Asteriaが拡張機能のバックエンドを読み込むときに使う正式な入口です。拡張機能だけで型を確認するときはExtension Kitの公開プロトコルを使い、Asteria上で実行するときはこの入口を使います。

## Custom Element（フルアクセス拡張）

フロントエンドは`asteria.extension.frontend.v1`契約の`asteriaExtension`を公開し、Vue 3とTypeScriptからビルドしたカスタム要素のESMファイルをエントリーポイントにします。

```ts
import { defineCustomElement, h } from 'vue'

const MyCustomElement = defineCustomElement({
  render() {
    return h('div', 'My extension')
  },
})

export const asteriaExtension = {
  contractVersion: 'asteria.extension.frontend.v1',
  elementName: 'asteria-com-example-my-extension',
  register() {
    if (!customElements.get(this.elementName)) {
      customElements.define(this.elementName, MyCustomElement)
    }
  },
}
```

ホストはカスタム要素に `data-asteria-extension-id`、`data-asteria-contribution-id`、`data-asteria-contribution-kind`、`data-asteria-contribution-surface` の4属性を付与します。画面遷移は`asteria:navigate`イベントの`detail`へ`{ target }`を入れて要求します。コア操作は`asteria:command`イベントの`detail`へ`{ id, payload }`を入れて要求します。どちらも`bubbles`と`composed`を有効にして発行し、ホストがデータ形式と許可範囲を検証します。

マニフェストのエントリーポイントに`.vue`ファイルを直接指定してはいけません。Extension Kitで`.mjs`にまとめてからパッケージへ入れてください。

## テストとビルド

作者のテストを先に実行し、成功したらビルドします。

```powershell
npm --prefix extensions/my-extension test
node extension-kit/scripts/build-extension.mjs --project extensions/my-extension
```

ビルダーはマニフェスト、必須ファイル、ライセンス、エントリーポイントを確認します。そのうえで、同じ入力から同じチェックサムになる `dist/<id>-<version>.zip` を生成します。

ビルダーは `setup.py`、パッケージスクリプト、公開処理、依存関係解決を実行しません。`npm install` や `pip install` もパッケージ導入時に実行しない契約です。必要な実行時ファイルは作者のビルド工程でパッケージへ含めます。

## 導入後の変更と再起動

導入、更新、有効化・無効化、前の版への復元、削除は、Asteriaを再起動した後に反映されます。更新は、利用者が**更新**を明示的に選んだ場合だけ実行します。Asteriaが管理する拡張機能は更新前の版を一世代保持するため、必要なら**元に戻す**を選べます。手動で指定したフォルダーは管理対象外で、Asteriaからの更新や復元はできません。

読み込みに失敗した拡張機能があっても、Asteria本体は起動します。通常起動できない場合は、ランチャーの**Safe Mode**で再起動してください。問題の拡張機能を無効化、再導入、または前の版へ戻してから、通常起動に戻します。**削除**は確認後に予約され、再起動後に反映されます。

## サンプル

標準拡張は、Extension Kitが作るひな形をそのまま最小例として利用できます。フルアクセス拡張の完成した構成は`examples/extensions/sample-full`を参照してください。マニフェスト、Pythonバックエンド、カスタム要素、ビルド設定、実在するMITライセンスがそろっています。配布前にID、表示名、バージョンなどを自分の値へ変更してください。
