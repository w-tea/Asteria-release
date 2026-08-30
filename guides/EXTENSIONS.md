# Extensions / 拡張機能

## 日本語

AsteriaのExtensionsでは、画面、パネル、操作を追加できます。拡張機能には、分離された画面と申告済みの仲介機能だけを使う**標準拡張**と、Asteria本体と同じ権限でコードを動かす**フルアクセス拡張**があります。

標準拡張は、PC内のファイルやAsteriaの内部データへ直接触れられません。入力中のプロンプトを読む場合や、外部サービスへ接続する場合は、導入前に権限と接続先を表示します。フルアクセス拡張は、ファイル、ネットワーク、外部プログラムへアクセスできます。導入前に実行方式、入手元、ライセンス、要求される権限を確認してください。

### 導入元と導入

導入後はExtensions画面から拡張機能を管理できます。まず**カタログ**、**URL / GitHub**、**ローカルZIP**、**拡張機能フォルダー**のいずれかを選び、名前、対応するAsteriaのバージョン、実行方式、ライセンス、要求される権限、チェックサムを確認します。

- **カタログ**には、Asteriaの公開カタログに掲載されたパッケージが表示されます。公式、コミュニティ、未確認の区分を確認できます。
- **URL / GitHub**では、HTTPSまたはGitHub ReleasesのZIPを指定します。期待するSHA-256を入力できます。未指定の場合は、未確認の導入元として扱われます。
- **ローカルZIP**では、PCに保存した拡張機能のZIPを選びます。
- **拡張機能フォルダー**では、開発中のフォルダーを選びます。手動で指定したフォルダーは、Asteriaからの更新や前の版へ戻す操作の対象外です。

内容を確認して **Install** を確定すると、次回の再起動後に導入結果が反映されます。パッケージのinstall script、`npm install`、`pip install`は自動で実行しません。

### 更新・無効化・復旧

変更後の状態は、次の操作と再起動で適用されます。

- **Update** を押すと、明示的に選んだ更新だけを取得・適用します。通知を確認しただけでは更新されません。
- 管理対象の更新は前の版を一世代保持します。前の版へ戻す（Roll back）場合は操作を選び、再起動してください。
- **Disable / Enable** は次回の再起動後に反映されます。
- **Uninstall** は確認後に削除を予約し、再起動後に反映されます。
- 起動できない場合はランチャーの **Safe Mode** で再起動します。拡張機能を読み込まない状態で、問題の拡張機能を無効化、再導入、または前の版へ戻してください。その後、通常起動へ戻します。

Asteriaは第三者拡張機能やカタログ掲載物の品質、安全性、動作、継続提供を保証しません。Safe Mode、無効化、前の版へ戻す機能は、起動できない状態から復旧するためのものです。拡張機能がすでに行った通信やファイル操作は取り消せません。

### 開発者向け

Extension Kit、Manifest、Custom Element、Python登録方法は[拡張機能開発者ガイド（日本語）](extensions/DEVELOPER_GUIDE.ja.md)を参照してください。

## English

Asteria Extensions can add workspaces, panels, and actions. A **Standard extension** runs in a sandboxed frame and uses only declared broker operations. A **Full-access extension** runs code with the same user-level access as Asteria and can access files, networks, and external programs. Review the execution level, source, license, and requested access before installing.

### Sources

- **Catalog**: A listed package with an Official, Community, or Unverified label.
- **URL**: An HTTPS or GitHub release ZIP. You may provide an expected SHA-256; without one, Asteria requires an explicit unverified-source confirmation.
- **ZIP**: A local extension archive.
- **Folder**: A local development folder. Manual folders are not covered by managed update or rollback.

Use the **Catalog** or **Install** tab to review the name, supported Asteria versions, execution level, license, permissions, and checksum before confirming. Restart Asteria to apply installation changes. Asteria does not run package install scripts, `npm install`, or `pip install` automatically.

### Update, recovery, and removal

- Update detection is informational. A download happens only after you press **Update**.
- Managed updates retain one previous version for **Roll back**. Restart to apply the rollback.
- **Disable / Enable** changes apply after the next restart.
- **Uninstall** is confirmed first and takes effect after restart.
- If an extension prevents normal startup, restart with the launcher’s **Safe Mode**. Safe Mode loads no extension code, allowing you to disable, reinstall, or roll back the problem extension before returning to normal startup.

Asteria does not guarantee the security, quality, operation, or continued availability of third-party extensions or catalog listings. Safe Mode, disable, and rollback are startup-recovery tools; they cannot undo network or file activity that already occurred.

For authors, see the [English Extension Developer Guide](extensions/DEVELOPER_GUIDE.md).
