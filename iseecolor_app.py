import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, 
                             QVBoxLayout, QHBoxLayout, QFrame, QSlider, QTableWidget, QTableWidgetItem, 
                             QGroupBox, QGridLayout, QStatusBar,QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QTimer, QThreadPool, QRunnable, pyqtSlot, QObject, pyqtSignal

from ultralytics import YOLO

class WorkerSignals(QObject):
  result = pyqtSignal(object, object, object)
  update = pyqtSignal(str)

class VideoProcessor(QRunnable):
  def __init__(self, cap, model, fixation_data):
      super().__init__()
      self.signals = WorkerSignals()
      self.cap = cap
      self.model = model
      self.fixation_data = fixation_data
      self.is_running = True

  @pyqtSlot()
  def run(self):
      while self.is_running:
          ret, frame = self.cap.read()
          if not ret:
              break
          
          frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
          results = self.model(frame, task='segment')
          frame_fixations = self.fixation_data[self.fixation_data['Frame'] == frame_number]
          
          self.signals.result.emit(frame, results, frame_fixations)
          self.signals.update.emit(f"Processing frame: {frame_number}")

  def stop(self):
      self.is_running = False

class ISeeColorApp(QMainWindow):
  def __init__(self):
      super().__init__()
      self.setWindowTitle("ISeeColor Visualization")
      self.setGeometry(100, 100, 1200, 800)
      
      self.model = YOLO('yolov8n-seg.pt')
      self.cap = None
      self.video_processor = None
      self.timer = QTimer(self)
      self.timer.timeout.connect(self.update_frame)
      
      self.fixation_data = None
      self.fixations_per_object = pd.DataFrame(columns=['Object', 'Fixation_Count'])
      self.object_counts = {}
      
      # Initialize master DataFrame
      self.master_fixation_data = pd.DataFrame(columns=['Frame', 'Object', 'Fixation_Count'])
      
      self.initUI()
      
      self.threadpool = QThreadPool()

  def initUI(self):
    self.setStyleSheet("background-color: #f5f5f5;")

    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    
    # Create a QSplitter to make the panes adjustable
    main_splitter = QSplitter(Qt.Horizontal, central_widget)

    # Left pane
    left_pane = QFrame()
    left_pane.setFrameShape(QFrame.StyledPanel)
    left_layout = QVBoxLayout(left_pane)
    left_layout.setSpacing(10)

    group_box_style = """
    QGroupBox {
        font-weight: bold;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 5px;
        margin-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
    }
    """

    # Video Information Group
    info_group = QGroupBox("Video Information")
    info_group.setStyleSheet(group_box_style)
    info_layout = QVBoxLayout()
    self.total_time_label = QLabel("Total Time (s): 0")
    self.total_frames_label = QLabel("Total Frames: 0")
    self.total_fixations_label = QLabel("Total Fixations: 0")
    self.object_interest_label = QLabel("Object of Interest: None")
    label_style = "font-size: 14px; color: #333;"
    for label in [self.total_time_label, self.total_frames_label, self.total_fixations_label, self.object_interest_label]:
        label.setStyleSheet(label_style)
        info_layout.addWidget(label)
    info_group.setLayout(info_layout)
    left_layout.addWidget(info_group)

    # Current Frame Objects Table
    current_frame_group = QGroupBox("Objects in Current Frame")
    current_frame_group.setStyleSheet(group_box_style)
    current_frame_layout = QVBoxLayout()
    self.current_frame_table = QTableWidget()
    self.current_frame_table.setColumnCount(2)
    self.current_frame_table.setHorizontalHeaderLabels(["Object", "Count"])
    table_header_style = """
    QHeaderView::section {
        padding: 4px;
        border: 1px solid #ddd;
    }
    """
    self.current_frame_table.horizontalHeader().setStyleSheet(table_header_style)
    self.current_frame_table.horizontalHeader().setStretchLastSection(True)
    self.current_frame_table.verticalHeader().setVisible(False)
    current_frame_layout.addWidget(self.current_frame_table)
    current_frame_group.setLayout(current_frame_layout)
    left_layout.addWidget(current_frame_group)

    # Fixated Objects Table
    fixated_objects_group = QGroupBox("Fixated Objects (Total)")
    fixated_objects_group.setStyleSheet(group_box_style)
    fixated_objects_layout = QVBoxLayout()
    self.fixated_objects_table = QTableWidget()
    self.fixated_objects_table.setColumnCount(2)
    self.fixated_objects_table.setHorizontalHeaderLabels(["Object", "Total Fixations"])
    self.fixated_objects_table.horizontalHeader().setStyleSheet(table_header_style)
    self.fixated_objects_table.horizontalHeader().setStretchLastSection(True)
    self.fixated_objects_table.verticalHeader().setVisible(False)
    fixated_objects_layout.addWidget(self.fixated_objects_table)
    fixated_objects_group.setLayout(fixated_objects_layout)
    left_layout.addWidget(fixated_objects_group)

    # Add left pane to splitter
    main_splitter.addWidget(left_pane)

    # Center pane
    center_pane = QFrame()
    center_pane.setFrameShape(QFrame.StyledPanel)
    center_layout = QVBoxLayout(center_pane)

    # Load Video and Load Fixation Data buttons
    load_buttons_layout = QHBoxLayout()
    self.load_video_button = QPushButton("Load Video")
    self.load_video_button.setIcon(QIcon.fromTheme("video-x-generic"))
    self.load_video_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")
    self.load_fixation_button = QPushButton("Load Fixation Data")
    self.load_fixation_button.setIcon(QIcon.fromTheme("text-x-generic"))
    self.load_fixation_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")
    load_buttons_layout.addWidget(self.load_video_button)
    load_buttons_layout.addWidget(self.load_fixation_button)
    center_layout.addLayout(load_buttons_layout)

    # Video Display
    self.video_display = QLabel()
    self.video_display.setStyleSheet("background-color: black; border: 1px solid #ccc; border-radius: 5px;")
    center_layout.addWidget(self.video_display)

    # Control Buttons
    controls_layout = QHBoxLayout()
    self.play_button = QPushButton("Play")
    self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
    self.play_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")
    self.pause_button = QPushButton("Pause")
    self.pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))
    self.pause_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")
    self.stop_button = QPushButton("Stop")
    self.stop_button.setIcon(QIcon.fromTheme("media-playback-stop"))
    self.stop_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")
    self.prev_frame_button = QPushButton("Previous Frame")
    self.prev_frame_button.setIcon(QIcon.fromTheme("go-previous"))
    self.prev_frame_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")
    self.next_frame_button = QPushButton("Next Frame")
    self.next_frame_button.setIcon(QIcon.fromTheme("go-next"))
    self.next_frame_button.setStyleSheet("background-color: #5DADE2; color: white; border: none; padding: 10px; border-radius: 5px;")

    for button in [self.play_button, self.pause_button, self.stop_button, self.prev_frame_button, self.next_frame_button]:
        controls_layout.addWidget(button)

    center_layout.addLayout(controls_layout)

    # Slider
    self.slider = QSlider(Qt.Horizontal)
    self.slider.setStyleSheet("""
    QSlider::groove:horizontal {
        border: 1px solid #999999;
        height: 8px;
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
        margin: 2px 0;
    }
    QSlider::handle:horizontal {
        background: #5DADE2;
        border: 1px solid #5c5c5c;
        width: 18px;
        margin: -2px 0;
        border-radius: 3px;
    }
    """)
    center_layout.addWidget(self.slider)

    # Frame Counter Label
    self.frame_counter_label = QLabel("Frame: 0 / 0")
    self.frame_counter_label.setAlignment(Qt.AlignCenter)
    self.frame_counter_label.setStyleSheet("font-size: 14px; color: #333;")
    center_layout.addWidget(self.frame_counter_label)

    # Add center pane to splitter
    main_splitter.addWidget(center_pane)

    # Set initial sizes for the splitter panes
    main_splitter.setSizes([400, 800])  # Adjust the initial sizes as needed

    # Add splitter to the main layout
    main_layout = QVBoxLayout(central_widget)
    main_layout.addWidget(main_splitter)

    # Connect buttons to functions
    self.load_video_button.clicked.connect(self.load_video)
    self.load_fixation_button.clicked.connect(self.load_fixation_data)
    self.play_button.clicked.connect(lambda: self.handle_button_click(self.play_button, self.play_video))
    self.pause_button.clicked.connect(lambda: self.handle_button_click(self.pause_button, self.pause_video))
    self.stop_button.clicked.connect(lambda: self.handle_button_click(self.stop_button, self.stop_video))
    self.next_frame_button.clicked.connect(lambda: self.handle_button_click(self.next_frame_button, self.next_frame))
    self.prev_frame_button.clicked.connect(lambda: self.handle_button_click(self.prev_frame_button, self.prev_frame))
    self.slider.sliderMoved.connect(self.slider_moved)

    # Add status bar
    self.statusBar = QStatusBar()
    self.setStatusBar(self.statusBar)
    self.statusBar.setStyleSheet("background-color: #f5f5f5; color: #333;")
    self.statusBar.showMessage("Ready")


  def reset_button_styles(self):
    button_style = """
    QPushButton {
        background-color: #5DADE2;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    """
    self.play_button.setStyleSheet(button_style)
    self.pause_button.setStyleSheet(button_style)
    self.stop_button.setStyleSheet(button_style)
    self.prev_frame_button.setStyleSheet(button_style)
    self.next_frame_button.setStyleSheet(button_style)

  def highlight_button(self, button):
    self.reset_button_styles()
    button.setStyleSheet("""
    QPushButton {
        background-color: #3498DB;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    """)

  def handle_button_click(self, button, action):
    self.highlight_button(button)
    action()

  def update_current_frame_table(self):
    self.current_frame_table.setRowCount(len(self.object_counts))
    for row, (obj, count) in enumerate(self.object_counts.items()):
        self.current_frame_table.setItem(row, 0, QTableWidgetItem(obj))
        self.current_frame_table.setItem(row, 1, QTableWidgetItem(str(count)))
    self.current_frame_table.sortItems(1, Qt.DescendingOrder)

  def update_fixated_objects_table(self):
      self.fixated_objects_table.setRowCount(len(self.fixations_per_object))
      for row, (obj, count) in enumerate(self.fixations_per_object.values):
          self.fixated_objects_table.setItem(row, 0, QTableWidgetItem(obj))
          self.fixated_objects_table.setItem(row, 1, QTableWidgetItem(str(count)))
      self.fixated_objects_table.sortItems(1, Qt.DescendingOrder)

  def update_fixated_objects_table_from_master(self, frame_number):
      # Filter the master dataframe for frames up to the current frame
      frame_data = self.master_fixation_data[self.master_fixation_data['Frame'] <= frame_number]
      
      # Group by object and sum the fixation counts, excluding empty objects
      grouped_data = frame_data[frame_data['Object'] != ''].groupby('Object')['Fixation_Count'].sum().reset_index()
      
      # Update the fixated objects table
      self.fixated_objects_table.setRowCount(len(grouped_data))
      for row, (obj, count) in enumerate(grouped_data.values):
          self.fixated_objects_table.setItem(row, 0, QTableWidgetItem(obj))
          self.fixated_objects_table.setItem(row, 1, QTableWidgetItem(str(count)))
      self.fixated_objects_table.sortItems(1, Qt.DescendingOrder)

  def clear_tables(self):
      self.current_frame_table.setRowCount(0)
      self.fixated_objects_table.setRowCount(0)

  def reset_tables(self):
      self.fixations_per_object = pd.DataFrame(columns=['Object', 'Fixation_Count'])
      self.object_counts = {}
      self.update_current_frame_table()
      self.update_fixated_objects_table()

  def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_time = total_frames / fps
            
            self.total_frames_label.setText(f"Total Frames: {total_frames}")
            self.total_time_label.setText(f"Total Time (s): {total_time:.2f}")
            self.slider.setMaximum(total_frames)
            self.frame_counter_label.setText(f"Frame: 0 / {total_frames}")
            
            # Initialize master DataFrame with all frame numbers
            self.initialize_master_dataframe(total_frames)

  def initialize_master_dataframe(self, total_frames):
      # Create a DataFrame with all frame numbers
      self.master_fixation_data = pd.DataFrame({
          'Frame': range(1, total_frames + 1),
          'Object': '',
          'Fixation_Count': 0
      })

  def load_fixation_data(self):
      file_name, _ = QFileDialog.getOpenFileName(self, "Open Fixation Data File", "", "CSV Files (*.csv)")
      if file_name:
          self.fixation_data = pd.read_csv(file_name)
          self.total_fixations_label.setText(f"Total Fixations: {len(self.fixation_data)}")

  def play_video(self):
      if self.cap is None:
          return
      self.timer.start(30)  # Update every 30 ms

  def pause_video(self):
      self.timer.stop()

  def stop_video(self):
      self.timer.stop()
      if self.cap:
          self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      self.update_frame()
      self.clear_tables()
      self.reset_tables()
      # Reinitialize the master DataFrame when stopping the video
      total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.initialize_master_dataframe(total_frames)

#   def jump_to_frame(self):
#       if self.cap is None:
#           return
#       frame_num = int(self.jump_input.text())
#       self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
#       self.update_frame()

  def next_frame(self):
      if self.cap is None:
          return
      current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
      self.update_frame()

  def prev_frame(self):
      if self.cap is None:
          return
      current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(current_frame - 2, 0))
      self.update_frame()

  def slider_moved(self, position):
      if self.cap is None:
          return
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
      total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
      self.frame_counter_label.setText(f"Frame: {position} / {total_frames}")
      self.update_frame()

  def update_frame(self):
    if self.cap is None or self.fixation_data is None:
        return
    ret, frame = self.cap.read()
    if not ret:
        self.timer.stop()
        return

    frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Resize the frame for faster YOLO inference
    scaling_factor = 0.5  # Adjust this factor as needed

    # Calculate the new dimensions based on the scaling factor
    new_width = int(frame.shape[1] * scaling_factor)
    new_height = int(frame.shape[0] * scaling_factor)

    # Resize the frame using the calculated dimensions
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    results = self.model(resized_frame, task='segment')
    
    frame_fixations = self.fixation_data[self.fixation_data['Frame'] == frame_number]
    
    self.process_fixations(results, frame_fixations, frame_number, frame.shape[:-1] )
    self.display_frame(frame, results)
    
    self.slider.setValue(frame_number)
    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.frame_counter_label.setText(f"Frame: {frame_number} / {total_frames}")
    
    # Update the fixated objects table based on the current frame
    self.update_fixated_objects_table_from_master(frame_number)

  def process_fixations(self, results, frame_fixations, frame_number, frame_dim):
        # Reset fixations for this frame
    self.master_fixation_data.loc[self.master_fixation_data['Frame'] == frame_number, ['Object', 'Fixation_Count']] = ['', 0]
    orig_height, orig_width = frame_dim[0], frame_dim[1]
    
    fixation_overlapped = False
    
    for _, fixation in frame_fixations.iterrows():
        x, y = fixation['X_Coordinate'], fixation['Y_Coordinate']
        
        # Get the dimensions of the original image and the segmentation mask
        
        mask_height, mask_width = results[0].masks.data.shape[1:]
        
        # Scale the fixation coordinates to match the mask dimensions
        scaled_x = int(x * mask_width / orig_width)
        scaled_y = int(y * mask_height / orig_height)
        
        # Ensure the scaled coordinates are within the mask bounds
        scaled_x = min(max(scaled_x, 0), mask_width - 1)
        scaled_y = min(max(scaled_y, 0), mask_height - 1)
        
        for seg, cls in zip(results[0].masks, results[0].boxes.cls):
            if seg.data[0, scaled_y, scaled_x] > 0:
                obj_class = results[0].names[int(cls)]
                self.object_interest_label.setText(f"Object of Interest: {obj_class}")
                fixation_overlapped = True
                
                # Update master DataFrame
                mask = (self.master_fixation_data['Frame'] == frame_number) & (self.master_fixation_data['Object'] == obj_class)
                if mask.any():
                    self.master_fixation_data.loc[mask, 'Fixation_Count'] += 1
                    print(self.master_fixation_data.loc[mask, 'Fixation_Count'])
                else:
                    self.master_fixation_data.loc[self.master_fixation_data['Frame'] == frame_number, ['Object', 'Fixation_Count']] = [obj_class, 1]
                
                break
    
    if not fixation_overlapped:
        self.object_interest_label.setText("Object of Interest: ")

    # Update the fixated objects table
    self.update_fixated_objects_table_from_master(frame_number)

  def display_frame(self, frame, results):
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480

    overlay = np.zeros_like(frame, dtype=np.uint8)
    
    frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_fixations = self.fixation_data[self.fixation_data['Frame'] == frame_number]
    
    # Reset object counts for this frame
    self.object_counts = {}
    
    if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        
        masks_resized = np.zeros((masks.shape[0], frame.shape[0], frame.shape[1]), dtype=bool)
        for i, mask in enumerate(masks):
            masks_resized[i] = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Count all objects in the frame
        for cls in results[0].boxes.cls:
            obj_class = results[0].names[int(cls)]
            self.object_counts[obj_class] = self.object_counts.get(obj_class, 0) + 1

        for _, fixation in frame_fixations.iterrows():
            x, y = int(fixation['X_Coordinate']), int(fixation['Y_Coordinate'])
            
            for mask, cls in zip(masks_resized, results[0].boxes.cls):
                if mask[y, x]:
                    obj_class = results[0].names[int(cls)]
                    
                    # Get total fixation count for this object type
                    total_fixations = self.master_fixation_data[
                        (self.master_fixation_data['Object'] == obj_class) & 
                        (self.master_fixation_data['Frame'] <= frame_number)
                    ]['Fixation_Count'].sum()
                    
                    # Determine color based on fixation count
                    if total_fixations <= 20:
                        color = (0, 255, 0)  # Green
                    elif total_fixations <= 40:
                        color = (255, 0, 0)  # Blue
                    elif total_fixations <= 60:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                    
                    overlay[mask] = color
                    break

        # Update the current frame table with all objects
        self.update_current_frame_table()

    combined_frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)
    
    # Draw unique fixation points with two colors
    for _, fixation in frame_fixations.iterrows():
        x, y = int(fixation['X_Coordinate']), int(fixation['Y_Coordinate'])
        # Draw a crosshair
        cv2.line(combined_frame, (x-10, y), (x+10, y), (255, 255, 255), 20)  # White
        cv2.line(combined_frame, (x, y-10), (x, y+10), (255, 255, 255), 20)  # White
        cv2.line(combined_frame, (x-10, y), (x+10, y), (0, 0, 0), 5)  # Black
        cv2.line(combined_frame, (x, y-10), (x, y+10), (0, 0, 0), 5)  # Black
        # Draw a circle around the crosshair
        cv2.circle(combined_frame, (x, y), 15, (255, 255, 255), 20)  # White
        cv2.circle(combined_frame, (x, y), 15, (0, 0, 0), 5)  # Black

    # Add legend
    legend_height = 30
    legend_frame = np.zeros((legend_height, combined_frame.shape[1], 3), dtype=np.uint8)
    
    # Define color ranges and labels
    color_ranges = [
        ((0, 255, 0), "1-20"),
        ((255, 0, 0), "21-40"),
        ((0, 255, 255), "41-60"),
        ((0, 0, 255), "61+")
    ]
    
    # Draw color boxes and labels
    box_width = combined_frame.shape[1] // len(color_ranges)
    for i, (color, label) in enumerate(color_ranges):
        start_x = i * box_width
        end_x = (i + 1) * box_width
        cv2.rectangle(legend_frame, (start_x, 0), (end_x, legend_height), color, -1)
        cv2.putText(legend_frame, label, (start_x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combine frame with legend
    combined_frame = np.vstack((combined_frame, legend_frame))

    # Resize and center the frame
    h, w = combined_frame.shape[:2]
    aspect_ratio = w / h
    if DISPLAY_WIDTH / DISPLAY_HEIGHT > aspect_ratio:
        new_height = DISPLAY_HEIGHT
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = DISPLAY_WIDTH
        new_height = int(new_width / aspect_ratio)

    resized_frame = cv2.resize(combined_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    y_offset = (DISPLAY_HEIGHT - new_height) // 2
    x_offset = (DISPLAY_WIDTH - new_width) // 2
    display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

    rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    self.video_display.setPixmap(QPixmap.fromImage(qt_image))

    del frame, results, overlay, combined_frame, resized_frame, rgb_image, qt_image

if __name__ == "__main__":
  app = QApplication(sys.argv)
  app.setStyle("Fusion")  # Use Fusion style for a more modern look
  window = ISeeColorApp()
  window.show()
  sys.exit(app.exec_())