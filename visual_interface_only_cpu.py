import os
import time
import sys
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, QUrl, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QColor, QBrush, QIcon, QPixmap
from PyQt5.QtChart import QDateTimeAxis, QValueAxis, QSplineSeries, QChart
import torch
from UI.main_window import Ui_MainWindow
from detect_visual import YOLOPredict
from utils.datasets import img_formats

CODE_VER = "V2.0"
PREDICT_SHOW_TAB_INDEX = 0
REAL_TIME_PREDICT_TAB_INDEX = 1


class PredictDataHandlerThread(QThread):
    predict_message_trigger = pyqtSignal(str)

    def __init__(self, predict_model):
        super(PredictDataHandlerThread, self).__init__()
        self.running = False
        self.predict_model = predict_model

    def run(self):
        self.running = True
        over_time = 0
        while self.running:
            if self.predict_model.predict_info != "":
                self.predict_message_trigger.emit(self.predict_model.predict_info)
                self.predict_model.predict_info = ""
                over_time = 0
            time.sleep(0.01)
            over_time += 1

            if over_time > 100000:
                self.running = False


class PredictHandlerThread(QThread):
    def __init__(self, input_player, output_player, out_file_path, weight_path,
                 predict_info_plain_text_edit, predict_progress_bar, fps_label,
                 button_dict, input_tab, output_tab, input_image_label, output_image_label,
                 real_time_show_predict_flag):
        super(PredictHandlerThread, self).__init__()
        self.running = False

        self.predict_model = YOLOPredict(weight_path, out_file_path)
        self.output_predict_file = ""
        self.parameter_source = ''

        self.input_player = input_player
        self.output_player = output_player
        self.predict_info_plainTextEdit = predict_info_plain_text_edit
        self.predict_progressBar = predict_progress_bar
        self.fps_label = fps_label
        self.button_dict = button_dict
        self.input_tab = input_tab
        self.output_tab = output_tab
        self.input_image_label = input_image_label
        self.output_image_label = output_image_label

        self.real_time_show_predict_flag = real_time_show_predict_flag

        self.predict_data_handler_thread = PredictDataHandlerThread(self.predict_model)
        self.predict_data_handler_thread.predict_message_trigger.connect(self.add_messages)

    def run(self):
        self.predict_data_handler_thread.start()

        self.predict_progressBar.setValue(0)
        for item, button in self.button_dict.items():
            button.setEnabled(False)

        image_flag = os.path.splitext(self.parameter_source)[-1].lower() in img_formats
        qt_input = None
        qt_output = None

        if not image_flag and self.real_time_show_predict_flag:
            qt_input = self.input_image_label
            qt_output = self.output_image_label
            self.input_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)
            self.output_tab.setCurrentIndex(REAL_TIME_PREDICT_TAB_INDEX)

        with torch.no_grad():
            self.output_predict_file = self.predict_model.detect(self.parameter_source,
                                                                 qt_input=qt_input,
                                                                 qt_output=qt_output)

        if self.output_predict_file != "":
            self.input_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.parameter_source)))
            self.input_player.pause()

            self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.output_predict_file)))
            self.output_player.pause()

            self.input_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
            self.output_tab.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)

            for item, button in self.button_dict.items():
                if image_flag and item in ['play_pushButton', 'pause_pushButton']:
                    continue
                button.setEnabled(True)

    @pyqtSlot(str)
    def add_messages(self, message):
        if message != "":
            self.predict_info_plainTextEdit.appendPlainText(message)

            if ":" not in message:
                return

            split_message = message.split(" ")

            if "video" in message:
                percent = split_message[2][1:-1].split("/")
                value = int((int(percent[0]) / int(percent[1])) * 100)
                value = value if (int(percent[1]) - int(percent[0])) > 2 else 100
                self.predict_progressBar.setValue(value)
            else:
                self.predict_progressBar.setValue(100)

            second_count = 1 / float(split_message[-1][1:-2])
            self.fps_label.setText(f"--> {second_count:.1f} FPS")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, weight_path, out_file_path, real_time_show_predict_flag: bool, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Intelligent Monitoring System of Construction Site Software " + CODE_VER)
        self.showMaximized()

        self.import_media_pushButton.clicked.connect(self.import_media)
        self.start_predict_pushButton.clicked.connect(self.predict_button_click)
        self.open_predict_file_pushButton.clicked.connect(self.open_file_in_browser)

        self.play_pushButton.clicked.connect(self.play_pause_button_click)
        self.pause_pushButton.clicked.connect(self.play_pause_button_click)
        self.button_dict = dict()
        self.button_dict.update({"import_media_pushButton": self.import_media_pushButton,
                                 "start_predict_pushButton": self.start_predict_pushButton,
                                 "open_predict_file_pushButton": self.open_predict_file_pushButton,
                                 "play_pushButton": self.play_pushButton,
                                 "pause_pushButton": self.pause_pushButton,
                                 "real_time_checkBox": self.real_time_checkBox
                                 })

        self.input_player = QMediaPlayer()
        self.input_player.setVideoOutput(self.input_video_widget)
        self.input_player.positionChanged.connect(self.change_slide_bar)

        self.output_player = QMediaPlayer()
        self.output_player.setVideoOutput(self.output_video_widget)

        self.series = QSplineSeries()
        self.chart_init()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.draw_gpu_info_chart)
        self.timer.start(1000)

        self.video_length = 0
        self.out_file_path = out_file_path
        self.predict_handler_thread = PredictHandlerThread(self.input_player,
                                                           self.output_player,
                                                           self.out_file_path,
                                                           weight_path,
                                                           self.predict_info_plainTextEdit,
                                                           self.predict_progressBar,
                                                           self.fps_label,
                                                           self.button_dict,
                                                           self.input_media_tabWidget,
                                                           self.output_media_tabWidget,
                                                           self.input_real_time_label,
                                                           self.output_real_time_label,
                                                           real_time_show_predict_flag
                                                           )
        self.weight_label.setText(f" Using weight : ****** {Path(weight_path[0]).name} ******")

        self.gen_better_gui()

        self.media_source = ""
        self.predict_progressBar.setValue(0)

        self.real_time_checkBox.stateChanged.connect(self.real_time_checkbox_state_changed)
        self.real_time_checkBox.setChecked(real_time_show_predict_flag)
        self.real_time_check_state = self.real_time_checkBox.isChecked()

    def gen_better_gui(self):
        play_icon = QIcon()
        play_icon.addPixmap(QPixmap("./UI/icon/play.png"), QIcon.Normal, QIcon.Off)
        self.play_pushButton.setIcon(play_icon)

        play_icon = QIcon()
        play_icon.addPixmap(QPixmap("./UI/icon/pause.png"), QIcon.Normal, QIcon.Off)
        self.pause_pushButton.setIcon(play_icon)

        self.input_media_tabWidget.tabBar().hide()
        self.output_media_tabWidget.tabBar().hide()
        self.input_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)
        self.output_media_tabWidget.setCurrentIndex(PREDICT_SHOW_TAB_INDEX)

        self.input_real_time_label.setStyleSheet("QLabel{background:black}")
        self.output_real_time_label.setStyleSheet("QLabel{background:black}")

    def real_time_checkbox_state_changed(self):
        self.real_time_check_state = self.real_time_checkBox.isChecked()
        self.predict_handler_thread.real_time_show_predict_flag = self.real_time_check_state

    def chart_init(self):
        self.gpu_info_chart._chart = QChart()
        self.gpu_info_chart._chart.setBackgroundBrush(QBrush(QColor("#FFFFFF")))

        self.series.setName("GPU Utilization")
        self.gpu_info_chart._chart.addSeries(self.series)
        self.dtaxisX = QDateTimeAxis()
        self.vlaxisY = QValueAxis()
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        self.vlaxisY.setMin(0)
        self.vlaxisY.setMax(100)
        self.dtaxisX.setFormat("hh:mm:ss")
        self.dtaxisX.setTickCount(5)
        self.vlaxisY.setTickCount(10)
        self.dtaxisX.setTitleText("Time")
        self.vlaxisY.setTitleText("Percent")
        self.vlaxisY.setGridLineVisible(False)
        self.gpu_info_chart._chart.addAxis(self.dtaxisX, Qt.AlignBottom)
        self.gpu_info_chart._chart.addAxis(self.vlaxisY, Qt.AlignLeft)
        self.series.attachAxis(self.dtaxisX)
        self.series.attachAxis(self.vlaxisY)
        self.gpu_info_chart.setChart(self.gpu_info_chart._chart)

    def draw_gpu_info_chart(self):
        time_current = QDateTime.currentDateTime()
        self.dtaxisX.setMin(QDateTime.currentDateTime().addSecs(-300 * 1))
        self.dtaxisX.setMax(QDateTime.currentDateTime().addSecs(0))
        remove_count = 600
        if self.series.count() > remove_count:
            self.series.removePoints(0, self.series.count() - remove_count)
        gpu_info = get_gpu_info()
        yint = gpu_info[0].get("gpu_load")
        self.series.append(time_current.toMSecsSinceEpoch(), yint)

    def import_media(self):
        self.media_source = QFileDialog.getOpenFileUrl()[0]
        self.input_player.setMedia(QMediaContent(self.media_source))

        path_current = str(Path.cwd().joinpath("area_dangerous\1.jpg"))
        self.output_player.setMedia(QMediaContent(QUrl.fromLocalFile(path_current)))

        self.predict_handler_thread.parameter_source = self.media_source.toLocalFile()
        self.input_player.pause()

        image_flag = os.path.splitext(self.predict_handler_thread.parameter_source)[-1].lower() in img_formats
        for item, button in self.button_dict.items():
            if image_flag and item in ['play_pushButton', 'pause_pushButton']:
                button.setEnabled(False)
            else:
                button.setEnabled(True)

    def predict_button_click(self):
        self.predict_handler_thread.start()

    def change_slide_bar(self, position):
        self.video_length = self.input_player.duration() + 0.1
        self.video_horizontalSlider.setValue(round((position / self.video_length) * 100))
        self.video_percent_label.setText(str(round((position / self.video_length) * 100, 2)) + '%')

    @pyqtSlot()
    def play_pause_button_click(self):
        name = self.sender().objectName()

        if self.media_source == "":
            return

        if name == "play_pushButton":
            print("play")
            self.input_player.play()
            self.output_player.play()

        elif name == "pause_pushButton":
            self.input_player.pause()
            self.output_player.pause()

    @pyqtSlot()
    def open_file_in_browser(self):
        os.system(f"start explorer {self.out_file_path}")

    @pyqtSlot()
    def closeEvent(self, *args, **kwargs):
        print("Close")


def get_gpu_info():
    return [{"gpu_id": 0,
             "gpu_memoryTotal": 0,
             "gpu_memoryUsed": 0,
             "gpu_memoryUtil": 0,
             "gpu_load": 0}]


if __name__ == '__main__':
    app = QApplication(sys.argv)

    weight_root = Path.cwd().joinpath("weights")
    if not weight_root.exists():
        raise FileNotFoundError("weights not found !!!")

    weight_file = [item for item in weight_root.iterdir() if item.suffix == ".pt"]
    weight_root = [str(weight_file[0])]
    out_file_root = Path.cwd().joinpath(r'inference/output')
    out_file_root.parent.mkdir(exist_ok=True)
    out_file_root.mkdir(exist_ok=True)

    real_time_show_predict = True

    main_window = MainWindow(weight_root, out_file_root, real_time_show_predict)

    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon/icon.ico"), QIcon.Normal, QIcon.Off)
    main_window.setWindowIcon(icon)

    main_window.show()
    sys.exit(app.exec_())
