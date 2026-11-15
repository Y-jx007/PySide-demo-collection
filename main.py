# -*- coding: utf-8 -*-
"""Application launcher window."""
from launcher_widgets import *  # noqa: F401,F403

class ApplicationLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.active_processes = []   # 每个 app 一个独立子进程
        self.cards = []
        self.cards_by_path = {}
        self.category_widgets = []
        self.category_by_name = {}
        self._drag_card = None
        self._drag_source_cat = None
        self._drag_source_index = -1
        self._placeholder = None
        self._placeholder_cat = None
        self._placeholder_anim = None
        self._drag_done = False
        self._closing = False

        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_processes)
        self._poll_timer.start()

        self.setup_ui()
        self.setup_apps()

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.WindowDeactivate, QEvent.ApplicationDeactivate):
            for category in self.category_widgets:
                category.end_pan()
            return super().eventFilter(obj, event)
        # 卡片铺满整行，滚轮很容易落在卡片而不是页面滚动区上。
        # 统一在这里处理：竖滚 = 翻页，Shift/横滚 = 横向浏览这一卷胶片。
        if event.type() == QEvent.Wheel and isinstance(obj, QWidget):
            if (obj.window() is self
                    and QApplication.activeModalWidget() is None
                    and not isinstance(obj, QScrollBar)):
                if self._scroll_from_wheel(obj, event):
                    return True
        return super().eventFilter(obj, event)

    def _scroll_from_wheel(self, obj, event):
        delta_y = event.pixelDelta().y() or event.angleDelta().y()
        delta_x = event.pixelDelta().x() or event.angleDelta().x()
        want_horizontal = bool(event.modifiers() & Qt.ShiftModifier) or (
            abs(delta_x) > abs(delta_y)
        )
        if want_horizontal:
            category = self._category_for_widget(obj)
            if category is not None:
                bar = category.scroll_area.horizontalScrollBar()
                delta = delta_y or delta_x
                if delta and bar.maximum() > bar.minimum():
                    bar.setValue(bar.value() - delta)
                    event.accept()
                    return True
        bar = self.outer_scroll.verticalScrollBar()
        if delta_y and bar.maximum() > bar.minimum():
            bar.setValue(bar.value() - delta_y)
            event.accept()
            return True
        return False

    def _category_for_widget(self, widget):
        parent = widget
        while parent is not None and not isinstance(parent, CategoryWidget):
            parent = parent.parent()
        return parent if isinstance(parent, CategoryWidget) else None

    def setup_ui(self):
        self.setWindowTitle("绘图工具集")
        self.setMinimumSize(LAUNCHER_MIN_WIDTH, LAUNCHER_MIN_HEIGHT)
        self.resize(LAUNCHER_DEFAULT_WIDTH, LAUNCHER_DEFAULT_HEIGHT)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move((screen.width() - self.width()) // 2,
                  (screen.height() - self.height()) // 2)

        self.setStyleSheet(f"QMainWindow {{ background-color: {PAGE_BG}; }}")

        central = LauncherBackdrop()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        mm = LAUNCHER_MAIN_MARGINS
        main_layout.setContentsMargins(mm[0], mm[1], mm[2], mm[3])

        self.outer_scroll = QScrollArea()
        self.outer_scroll.setWidgetResizable(True)
        self.outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.outer_scroll.setFrameShape(QFrame.NoFrame)
        self.outer_scroll.viewport().setAutoFillBackground(False)
        self._style_outer_scrollbar()

        self.scroll_content = QWidget()
        self.scroll_content.setAttribute(Qt.WA_StyledBackground, True)
        # 胶片之间的纸面必须是浅色；不依赖上层 viewport 的默认 palette
        self.scroll_content.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {PAGE_TOP}, stop:1 {PAGE_BOTTOM});
        """)
        self.categories_layout = QVBoxLayout(self.scroll_content)
        self.categories_layout.setContentsMargins(0, 0, 0, 0)
        self.categories_layout.setSpacing(LAUNCHER_CATEGORY_SPACING)

        self.outer_scroll.setWidget(self.scroll_content)
        main_layout.addWidget(self.outer_scroll, 1)

        info = QLabel(INFO_DEFAULT)
        info.setFont(QFont(FAMILY_TITLE, INFO_FONT_SIZE))
        info.setStyleSheet(f"color: {PAGE_MUTED}; background: transparent; padding: 10px 0 6px 0;")
        info.setAlignment(Qt.AlignCenter)
        self.info_label = info
        main_layout.addWidget(info)

    def _style_outer_scrollbar(self):
        bar = self.outer_scroll.verticalScrollBar()
        bar.setStyleSheet("""
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 2px 0 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(92, 84, 72, 0.35);
                min-height: 40px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(92, 84, 72, 0.55);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)

    def _make_launcher(self, module_subpath, friendly_name):
        module_path = os.path.join(APPS_DIR, module_subpath.replace('/', os.sep))
        return lambda: self.launch_app(module_path, friendly_name)

    def _new_category(self, category_name):
        existing = self.category_by_name.get(category_name)
        if existing is not None:
            return existing
        cat_widget = CategoryWidget(category_name)
        cat_widget.card_dropped.connect(self._on_card_dropped)
        self.categories_layout.addWidget(cat_widget)
        self.category_widgets.append(cat_widget)
        self.category_by_name[category_name] = cat_widget
        return cat_widget

    def _add_card(self, cat_widget, app):
        if app["path"] in self.cards_by_path:
            return
        command = self._make_launcher(app["path"], app["friendly"])
        module_filename = os.path.basename(app["path"])
        image_name = os.path.splitext(module_filename)[0] + ".png"
        image_path = os.path.join(THUMBNAIL_DIR, image_name)
        card = AppCard(
            app["title"], app["description"], command,
            bg_image_path=image_path, app_path=app["path"],
        )
        self.cards.append(card)
        self.cards_by_path[app["path"]] = card
        cat_widget.add_card(card)

    def _create_category(self, category_name, items):
        """创建分类槽并添加卡片"""
        cat_widget = self._new_category(category_name)
        for app in items:
            self._add_card(cat_widget, app)
        return cat_widget

    def card_for_path(self, path):
        return self.cards_by_path.get(path)

    def _load_layout(self):
        try:
            with open(LAYOUT_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            categories = data.get("categories", [])
            if not isinstance(categories, list):
                return None
            out = []
            for entry in categories:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                apps = entry.get("apps", [])
                if isinstance(name, str) and isinstance(apps, list):
                    out.append((name, [p for p in apps if isinstance(p, str)]))
            return out or None
        except (OSError, ValueError):
            return None

    def _save_layout(self):
        data = {"version": 1, "categories": []}
        for cat in self.category_widgets:
            paths = []
            for i in range(cat.card_count()):
                item = cat.cards_layout.itemAt(i)
                card = item.widget() if item else None
                if card is not None and card.app_path:
                    paths.append(card.app_path)
            data["categories"].append({"name": cat.category_name, "apps": paths})
        tmp = LAYOUT_PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, LAYOUT_PATH)
        except OSError as exc:
            logging.warning(f"无法保存启动器布局: {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _make_placeholder(self):
        ph = QFrame()
        ph.setFixedSize(CARD_WIDTH, CARD_HEIGHT)
        ph.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ph.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(255, 248, 235, 20);
                border: 2px dashed rgba(255, 248, 235, 170);
                border-radius: {CARD_RADIUS}px;
            }}
        """)
        return ph

    def _animate_placeholder(self):
        if self._placeholder is None:
            return
        ph = self._placeholder
        if self._placeholder_anim is not None:
            self._placeholder_anim.stop()
        ph.setFixedWidth(0)
        anim = QVariantAnimation(self)
        anim.setDuration(130)
        anim.setStartValue(0)
        anim.setEndValue(CARD_WIDTH)
        anim.valueChanged.connect(lambda value: ph.setFixedWidth(int(value)))
        anim.finished.connect(lambda: ph.setFixedWidth(CARD_WIDTH))
        self._placeholder_anim = anim
        anim.start()

    def _remove_placeholder(self):
        if self._placeholder_anim is not None:
            self._placeholder_anim.stop()
            self._placeholder_anim = None
        if self._placeholder_cat is not None and self._placeholder is not None:
            self._placeholder_cat.remove_placeholder(self._placeholder)
            self._placeholder_cat = None

    def begin_card_drag(self, card):
        if self._drag_card is not None or card.category_widget is None:
            return False
        source = card.category_widget
        idx = source.cards_layout.indexOf(card)
        self._drag_card = card
        self._drag_source_cat = source
        self._drag_source_index = idx
        self._drag_done = False
        self._placeholder = self._make_placeholder()
        self._placeholder_cat = None
        card.hide()
        source.remove_card(card)
        self.info_label.setText(f"移动 {card.title}…")
        return True

    def update_drag_target(self, category, event):
        if self._drag_card is None or self._placeholder is None:
            return
        index = category._drop_index(event, getattr(category, "_drop_event_obj", None))
        if self._placeholder_cat is not category:
            if self._placeholder_cat is not None:
                self._placeholder_cat.remove_placeholder(self._placeholder)
            category.insert_placeholder(self._placeholder, index)
            self._placeholder_cat = category
            self._placeholder.show()
            self._animate_placeholder()
        else:
            if category.placeholder_card_index(self._placeholder) != index:
                category.remove_placeholder(self._placeholder)
                category.insert_placeholder(self._placeholder, index)
                self._placeholder.show()

    def leave_drag_target(self, category):
        if self._placeholder_cat is category:
            self._remove_placeholder()

    def _clear_card_drag(self):
        if self._placeholder_anim is not None:
            self._placeholder_anim.stop()
            self._placeholder_anim = None
        self._drag_card = None
        self._drag_source_cat = None
        self._drag_source_index = -1
        self._placeholder = None
        self._placeholder_cat = None
        self._drag_done = True

    def card_drag_finished(self):
        return self._drag_done

    def cancel_card_drag(self):
        if self._drag_card is None:
            return
        card = self._drag_card
        source = self._drag_source_cat
        index = self._drag_source_index
        self._remove_placeholder()
        if source is not None:
            source.insert_card(card, index)
        card.show()
        self._clear_card_drag()
        self._restore_info()

    def _relocate_app_file(self, card, target):
        """跨分类拖卡片时，把对应的 .py 挪到目标分类目录并同步清单。

        返回 False 表示这次拖动要放弃（文件动不了 / 正在运行 / 同名冲突）。
        """
        if not card.app_path or target is None:
            return True
        old_rel = card.app_path
        category = (target.category_name or "").strip()
        old_dir = old_rel.split("/")[0] if "/" in old_rel else ""
        if category == old_dir:
            return True
        if not category or category in (".", "..") or any(
                ch in category for ch in '\\/:*?"<>|'):
            self.show_error(f"分类名「{category}」不能用作文件夹名，文件保持原位。")
            return False

        name = os.path.basename(old_rel)
        new_rel = f"{category}/{name}"
        old_abs = os.path.join(APPS_DIR, old_rel.replace("/", os.sep))
        new_abs = os.path.join(APPS_DIR, new_rel.replace("/", os.sep))
        if not os.path.isfile(old_abs):
            self.show_error(f"找不到应用文件，卡片先留在原分类：\n{old_abs}")
            return False
        if os.path.exists(new_abs):
            self.show_error(f"目标分类里已经有同名文件：\n{new_abs}")
            return False
        for entry in self.active_processes:
            if (os.path.normcase(entry["path"]) == os.path.normcase(old_abs)
                    and entry["process"].poll() is None):
                self.show_error(f"{card.title} 正在运行，先关掉它再换分类。")
                return False

        try:
            os.makedirs(os.path.dirname(new_abs), exist_ok=True)
            shutil.move(old_abs, new_abs)
        except OSError as exc:
            self.show_error(f"移动文件失败：{exc}")
            return False
        if not rewrite_app_entry(old_rel, new_rel, category):
            try:
                shutil.move(new_abs, old_abs)
            except OSError:
                pass
            self.show_error("写回 apps_manifest.py 失败，文件已移回原处。")
            return False

        app = None
        for entry in APPS:
            if entry.get("path") == old_rel:
                entry["path"] = new_rel
                entry["category"] = category
                app = entry
                break
        for entry in self.active_processes:
            if os.path.normcase(entry["path"]) == os.path.normcase(old_abs):
                entry["path"] = new_abs
        self.cards_by_path.pop(old_rel, None)
        card.app_path = new_rel
        self.cards_by_path[new_rel] = card
        try:
            card.button.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        card.command = self._make_launcher(new_rel, (app or {}).get("friendly", card.title))
        card.button.clicked.connect(card._on_launch_clicked)
        return True

    def _finish_card_drag(self, target, index=None):
        if self._drag_card is None or target is None:
            self.cancel_card_drag()
            return
        card = self._drag_card
        if self._placeholder is not None and self._placeholder_cat is target:
            index = target.placeholder_card_index(self._placeholder)
        self._remove_placeholder()
        if not self._relocate_app_file(card, target):
            self.cancel_card_drag()
            return
        index = max(0, min(int(index or 0), target.card_count()))
        target.insert_card(card, index)
        card.show()
        self._save_layout()
        self.info_label.setText(f"已移动 {card.title} 到「{target.category_name}」")
        self._clear_card_drag()
        QTimer.singleShot(2200, self._restore_info)

    def _on_card_dropped(self, card, index):
        target = self.sender()
        if isinstance(target, CategoryWidget):
            self._finish_card_drag(target, index)

    def _move_card(self, card, target, index):
        # kept for programmatic moves/tests
        source = card.category_widget
        if source is None or target is None:
            return
        if not self._relocate_app_file(card, target):
            return
        source_index = source.cards_layout.indexOf(card)
        if source is target and 0 <= source_index < index:
            index -= 1
        source.remove_card(card)
        target.insert_card(card, index)
        card.category_widget = target
        self._save_layout()

    def setup_apps(self):
        app_map = {app["path"]: app for app in APPS}
        custom = self._load_layout()
        seen = set()
        if custom:
            for name, paths in custom:
                cat = self._new_category(name)
                for path in paths:
                    app = app_map.get(path)
                    if app is not None and path not in seen:
                        self._add_card(cat, app)
                        seen.add(path)

        remaining = [app for app in APPS if app["path"] not in seen]
        categories = {}
        for app in remaining:
            categories.setdefault(app["category"], []).append(app)

        # 优先按指定顺序显示；自定义布局里的分类保持原有顺序
        for cat_name in CATEGORY_ORDER:
            if cat_name in categories:
                cat = self._new_category(cat_name)
                for app in categories[cat_name]:
                    self._add_card(cat, app)

        for cat_name in sorted(categories.keys()):
            if cat_name in CATEGORY_ORDER:
                continue
            cat = self._new_category(cat_name)
            for app in categories[cat_name]:
                self._add_card(cat, app)

        self.categories_layout.addStretch()

    def _manifest_for_path(self, module_path):
        target = os.path.normcase(os.path.abspath(module_path))
        for app in APPS:
            candidate = os.path.normcase(os.path.abspath(
                os.path.join(APPS_DIR, app["path"].replace("/", os.sep))))
            if candidate == target:
                return app
        return None

    def _missing_dependencies(self, app):
        missing = []
        for dep in app.get("requires", []):
            try:
                if importlib.util.find_spec(dep) is None:
                    missing.append(dep)
            except (ImportError, ValueError):
                missing.append(dep)
        return missing

    def launch_app(self, module_path, friendly_name):
        if not os.path.exists(module_path):
            self.show_error(f"找不到应用文件:\n{module_path}")
            return False

        app = self._manifest_for_path(module_path)
        missing = self._missing_dependencies(app) if app else []
        if missing:
            hints = "\n".join(f"{dep}: {INSTALL_HINTS.get(dep, 'pip install ' + dep)}" for dep in missing)
            QMessageBox.warning(
                self, "缺少依赖",
                f"{friendly_name} 需要以下依赖：\n{hints}\n\n安装后再启动。"
            )
            return False

        for entry in self.active_processes:
            if entry["path"] == module_path and entry["process"].poll() is None:
                QMessageBox.information(self, "已在运行", f"{friendly_name} 已经在运行。")
                return "already"

        runner = os.path.join(PROJECT_DIR, "app_runner.py")
        if not os.path.exists(runner):
            self.show_error(f"找不到启动器入口:\n{runner}")
            return False

        # 子进程输出写入 runtime/logs：比管道更稳（打包后无控制台时也能查错）
        log_dir = os.path.join(RUNTIME_DIR, "logs")
        base = os.path.splitext(os.path.basename(module_path))[0]
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
        out_log = os.path.join(log_dir, f"{safe}.out.log")
        error_log = os.path.join(log_dir, f"{safe}.err.log")
        handles = []
        try:
            os.makedirs(log_dir, exist_ok=True)
            out_handle = open(out_log, "w", encoding="utf-8", errors="replace")
            handles.append(out_handle)
            err_handle = open(error_log, "w", encoding="utf-8", errors="replace")
            handles.append(err_handle)
        except OSError:
            for handle in handles:
                handle.close()
            handles = []
            out_handle = err_handle = subprocess.DEVNULL
            out_log = error_log = None

        creationflags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(
                [sys.executable, runner, module_path],
                cwd=PROJECT_DIR,
                stdout=out_handle,
                stderr=err_handle,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.show_error(f"无法启动 {friendly_name}:\n{exc}")
            return False
        finally:
            for handle in handles:
                handle.close()

        self.active_processes.append({
            "process": proc,
            "path": module_path,
            "friendly": friendly_name,
            "error_log": error_log,
        })
        self.info_label.setText(f"正在启动 {friendly_name}…（首次启动可能需要几秒）")
        QTimer.singleShot(6000, self._restore_info)
        return True

    def _poll_processes(self):
        changed = False
        for entry in list(self.active_processes):
            code = entry["process"].poll()
            if code is None:
                continue
            self.active_processes.remove(entry)
            changed = True
            if self._closing:
                continue
            if code != 0:
                hint = f"\n\n错误日志:\n{entry['error_log']}" if entry.get("error_log") else ""
                self.show_error(f"{entry['friendly']} 异常退出（代码 {code}）。{hint}")
        if changed:
            self._restore_info()

    def _restore_info(self):
        if self._closing:
            return
        if self.active_processes:
            self.info_label.setText(
                f"已启动 {len(self.active_processes)} 个工具 | 关闭启动器会一并关闭它们"
            )
        else:
            self.info_label.setText(INFO_DEFAULT)

    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        self._closing = True
        self._poll_timer.stop()
        for category in self.category_widgets:
            category.end_pan()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        entries = list(self.active_processes)
        for entry in entries:
            proc = entry["process"]
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        deadline = time.monotonic() + 3.0
        for entry in entries:
            proc = entry["process"]
            try:
                proc.wait(timeout=max(0.05, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
        self.active_processes.clear()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont(FAMILY_DEFAULT, 9))
    launcher = ApplicationLauncher()
    launcher.show()
    sys.exit(app.exec())