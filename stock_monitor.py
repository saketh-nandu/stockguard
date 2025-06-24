import asyncio
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import yfinance as yf
import pyttsx3
import webbrowser
from plyer import notification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataManager:
    """Handles stock data fetching and management"""
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 300  # 5 minutes
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch stock data with caching"""
        cache_key = f"{symbol}_{period}"
        
        if (cache_key in self.cache and 
            datetime.now() < self.cache_expiry.get(cache_key, datetime.min)):
            return self.cache[cache_key]
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if not data.empty:
                self.cache[cache_key] = data
                self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=self.cache_duration)
                return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
        
        return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

class AIPredictor:
    """AI model for stock price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction"""
        if len(data) < 5:
            return None
            
        # Technical indicators
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=min(20, len(data))).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'] = self.calculate_macd(data['Close'])
        
        # Price change features
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Select features
        feature_cols = self.feature_columns + ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'Price_Change', 'Volume_Change']
        features = data[feature_cols].fillna(0)
        
        return features.values
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        exp12 = prices.ewm(span=12).mean()
        exp26 = prices.ewm(span=26).mean()
        return exp12 - exp26
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """Train the AI model"""
        try:
            features = self.prepare_features(data)
            if features is None or len(features) < 10:
                return False
            
            # Prepare target (next day's closing price)
            targets = data['Close'].shift(-1).dropna().values
            features = features[:-1]  # Remove last row to match targets
            
            if len(features) != len(targets):
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Create and train model
            self.model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(features_scaled.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(1)
            ])
            
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            self.model.fit(features_scaled, targets, epochs=50, batch_size=8, verbose=0)
            self.is_trained = True
            
            logger.info("Model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Optional[Dict]:
        """Make price prediction"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            features = self.prepare_features(data)
            if features is None:
                return None
            
            # Use last available data point
            last_features = features[-1:].reshape(1, -1)
            features_scaled = self.scaler.transform(last_features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            current_price = data['Close'].iloc[-1]
            
            # Calculate confidence (simplified)
            price_volatility = data['Close'].pct_change().std()
            confidence = max(0.5, min(0.95, 1 - price_volatility * 10))
            
            return {
                'predicted_price': float(prediction),
                'current_price': float(current_price),
                'price_change': float(prediction - current_price),
                'percentage_change': float((prediction - current_price) / current_price * 100),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

class DatabaseManager:
    """Manages SQLite database for storing predictions and performance"""
    
    def __init__(self, db_path: str = "stockguard.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    price_change REAL NOT NULL,
                    percentage_change REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    actual_price REAL DEFAULT NULL,
                    accuracy REAL DEFAULT NULL
                )
            ''')
            
            # User settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def save_prediction(self, symbol: str, prediction: Dict):
        """Save prediction to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (symbol, predicted_price, current_price, price_change, percentage_change, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                prediction['predicted_price'],
                prediction['current_price'],
                prediction['price_change'],
                prediction['percentage_change'],
                prediction['confidence'],
                prediction['timestamp']
            ))
            conn.commit()
    
    def get_predictions(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """Get predictions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                    SELECT * FROM predictions 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

class NotificationManager:
    """Handles various types of notifications"""
    
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.voice_enabled = True
        self.desktop_enabled = True
    
    def show_desktop_notification(self, title: str, message: str):
        """Show desktop notification"""
        if self.desktop_enabled:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="StockGuard",
                    timeout=5
                )
            except Exception as e:
                logger.error(f"Error showing desktop notification: {e}")
    
    def speak_notification(self, text: str):
        """Speak notification using TTS"""
        if self.voice_enabled:
            try:
                def speak():
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                
                # Run TTS in separate thread to avoid blocking
                threading.Thread(target=speak, daemon=True).start()
            except Exception as e:
                logger.error(f"Error with voice notification: {e}")
    
    def send_prediction_alert(self, symbol: str, prediction: Dict):
        """Send prediction alert via multiple channels"""
        change_pct = prediction['percentage_change']
        confidence = prediction['confidence']
        
        # Format message
        direction = "UP" if change_pct > 0 else "DOWN"
        message = f"{symbol}: {direction} {abs(change_pct):.2f}% (Confidence: {confidence:.0%})"
        
        # Desktop notification
        self.show_desktop_notification("StockGuard Prediction", message)
        
        # Voice notification for significant changes
        if abs(change_pct) > 2.0:
            voice_message = f"Stock alert: {symbol} predicted to move {direction} by {abs(change_pct):.1f} percent"
            self.speak_notification(voice_message)

class StockGuardGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("StockGuard - Intelligent Stock Predictions")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize components
        self.data_manager = StockDataManager()
        self.ai_predictor = AIPredictor()
        self.db_manager = DatabaseManager()
        self.notification_manager = NotificationManager()
        
        # State variables
        self.monitored_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        self.is_monitoring = False
        self.monitoring_thread = None
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create main style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#1e1e1e', foreground='#ffffff')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#2d2d2d', foreground='#ffffff')
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="StockGuard - AI Stock Predictions", 
                              font=('Arial', 20, 'bold'), bg='#1e1e1e', fg='#4CAF50')
        title_label.pack(pady=(0, 20))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop monitoring
        self.monitor_button = tk.Button(control_frame, text="Start Monitoring", 
                                       command=self.toggle_monitoring,
                                       bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'),
                                       relief=tk.FLAT, padx=20, pady=5)
        self.monitor_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Add stock entry
        tk.Label(control_frame, text="Add Stock:", bg='#2d2d2d', fg='white').pack(side=tk.LEFT, padx=(20, 5), pady=10)
        self.stock_entry = tk.Entry(control_frame, font=('Arial', 10))
        self.stock_entry.pack(side=tk.LEFT, padx=5, pady=10)
        self.stock_entry.bind('<Return>', self.add_stock)
        
        add_button = tk.Button(control_frame, text="Add", command=self.add_stock,
                              bg='#2196F3', fg='white', relief=tk.FLAT)
        add_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Status: Stopped", 
                                    bg='#2d2d2d', fg='#ffcc00', font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Dashboard tab
        self.create_dashboard_tab(notebook)
        
        # Predictions tab
        self.create_predictions_tab(notebook)
        
        # Settings tab
        self.create_settings_tab(notebook)
    
    def create_dashboard_tab(self, notebook):
        """Create the main dashboard tab"""
        dashboard_frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(dashboard_frame, text="Dashboard")
        
        # Live predictions frame
        predictions_frame = tk.LabelFrame(dashboard_frame, text="Live Predictions", 
                                         bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        predictions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview for live data
        columns = ('Symbol', 'Current Price', 'Predicted Price', 'Change %', 'Confidence', 'Time')
        self.live_tree = ttk.Treeview(predictions_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.live_tree.heading(col, text=col)
            self.live_tree.column(col, width=120, anchor='center')
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(predictions_frame, orient=tk.VERTICAL, command=self.live_tree.yview)
        self.live_tree.configure(yscrollcommand=scrollbar.set)
        
        self.live_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
    
    def create_predictions_tab(self, notebook):
        """Create the predictions history tab"""
        pred_frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(pred_frame, text="Prediction History")
        
        # Filter frame
        filter_frame = tk.Frame(pred_frame, bg='#2d2d2d')
        filter_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(filter_frame, text="Filter by Stock:", bg='#2d2d2d', fg='white').pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar()
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, values=['All'] + self.monitored_stocks)
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind('<<ComboboxSelected>>', self.filter_predictions)
        
        refresh_button = tk.Button(filter_frame, text="Refresh", command=self.load_prediction_history,
                                  bg='#4CAF50', fg='white', relief=tk.FLAT)
        refresh_button.pack(side=tk.LEFT, padx=10)
        
        # History treeview
        hist_columns = ('Symbol', 'Predicted', 'Actual', 'Change %', 'Confidence', 'Accuracy', 'Date')
        self.history_tree = ttk.Treeview(pred_frame, columns=hist_columns, show='headings', height=20)
        
        for col in hist_columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100, anchor='center')
        
        hist_scrollbar = ttk.Scrollbar(pred_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=hist_scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        hist_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
    
    def create_settings_tab(self, notebook):
        """Create the settings tab"""
        settings_frame = tk.Frame(notebook, bg='#1e1e1e')
        notebook.add(settings_frame, text="Settings")
        
        # Notification settings
        notif_frame = tk.LabelFrame(settings_frame, text="Notification Settings", 
                                   bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        notif_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.voice_var = tk.BooleanVar(value=True)
        voice_check = tk.Checkbutton(notif_frame, text="Enable Voice Notifications", 
                                    variable=self.voice_var, bg='#2d2d2d', fg='white',
                                    command=self.update_notification_settings)
        voice_check.pack(anchor=tk.W, padx=10, pady=5)
        
        self.desktop_var = tk.BooleanVar(value=True)
        desktop_check = tk.Checkbutton(notif_frame, text="Enable Desktop Notifications", 
                                      variable=self.desktop_var, bg='#2d2d2d', fg='white',
                                      command=self.update_notification_settings)
        desktop_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # Monitoring settings
        monitor_frame = tk.LabelFrame(settings_frame, text="Monitoring Settings", 
                                     bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        monitor_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(monitor_frame, text="Update Interval (seconds):", bg='#2d2d2d', fg='white').pack(anchor=tk.W, padx=10, pady=5)
        self.interval_var = tk.StringVar(value="300")
        interval_entry = tk.Entry(monitor_frame, textvariable=self.interval_var)
        interval_entry.pack(anchor=tk.W, padx=10, pady=5)
        
        # Watched stocks
        stocks_frame = tk.LabelFrame(settings_frame, text="Monitored Stocks", 
                                    bg='#2d2d2d', fg='white', font=('Arial', 12, 'bold'))
        stocks_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.stocks_listbox = tk.Listbox(stocks_frame, bg='#3d3d3d', fg='white')
        self.stocks_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        remove_button = tk.Button(stocks_frame, text="Remove Selected", 
                                 command=self.remove_selected_stock,
                                 bg='#f44336', fg='white', relief=tk.FLAT)
        remove_button.pack(pady=5)
        
        self.update_stocks_listbox()
    
    def toggle_monitoring(self):
        """Start or stop monitoring"""
        if not self.is_monitoring:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring process"""
        self.is_monitoring = True
        self.monitor_button.config(text="Stop Monitoring", bg='#f44336')
        self.status_label.config(text="Status: Running", fg='#4CAF50')
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        self.monitor_button.config(text="Start Monitoring", bg='#4CAF50')
        self.status_label.config(text="Status: Stopped", fg='#ffcc00')
        logger.info("Monitoring stopped")
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                for symbol in self.monitored_stocks:
                    if not self.is_monitoring:
                        break
                    
                    # Get stock data
                    data = self.data_manager.get_stock_data(symbol, "3mo")
                    if data is None or len(data) < 30:
                        continue
                    
                    # Train model if not trained
                    if not self.ai_predictor.is_trained:
                        self.ai_predictor.train_model(data)
                    
                    # Make prediction
                    prediction = self.ai_predictor.predict(data)
                    if prediction:
                        # Save prediction
                        self.db_manager.save_prediction(symbol, prediction)
                        
                        # Update GUI
                        self.root.after(0, self.update_live_predictions, symbol, prediction)
                        
                        # Send notifications for significant changes
                        if abs(prediction['percentage_change']) > 1.0:
                            self.notification_manager.send_prediction_alert(symbol, prediction)
                
                # Wait for next update
                interval = int(self.interval_var.get())
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def update_live_predictions(self, symbol: str, prediction: Dict):
        """Update the live predictions display"""
        # Find existing item or create new one
        for item in self.live_tree.get_children():
            if self.live_tree.item(item)['values'][0] == symbol:
                self.live_tree.delete(item)
                break
        
        # Add new prediction
        values = (
            symbol,
            f"${prediction['current_price']:.2f}",
            f"${prediction['predicted_price']:.2f}",
            f"{prediction['percentage_change']:+.2f}%",
            f"{prediction['confidence']:.0%}",
            datetime.fromisoformat(prediction['timestamp']).strftime("%H:%M:%S")
        )
        
        # Color coding based on prediction
        item = self.live_tree.insert('', 0, values=values)
        if prediction['percentage_change'] > 0:
            self.live_tree.item(item, tags=('positive',))
        else:
            self.live_tree.item(item, tags=('negative',))
        
        # Configure tags
        self.live_tree.tag_configure('positive', background='#2d5a2d')
        self.live_tree.tag_configure('negative', background='#5a2d2d')
    
    def add_stock(self, event=None):
        """Add a new stock to monitoring"""
        symbol = self.stock_entry.get().strip().upper()
        if symbol and symbol not in self.monitored_stocks:
            # Validate symbol
            if self.data_manager.get_current_price(symbol):
                self.monitored_stocks.append(symbol)
                self.update_stocks_listbox()
                self.stock_entry.delete(0, tk.END)
                logger.info(f"Added {symbol} to monitoring")
            else:
                messagebox.showerror("Error", f"Invalid stock symbol: {symbol}")
    
    def remove_selected_stock(self):
        """Remove selected stock from monitoring"""
        selection = self.stocks_listbox.curselection()
        if selection:
            symbol = self.stocks_listbox.get(selection[0])
            self.monitored_stocks.remove(symbol)
            self.update_stocks_listbox()
            logger.info(f"Removed {symbol} from monitoring")
    
    def update_stocks_listbox(self):
        """Update the stocks listbox"""
        self.stocks_listbox.delete(0, tk.END)
        for stock in self.monitored_stocks:
            self.stocks_listbox.insert(tk.END, stock)
    
    def load_prediction_history(self):
        """Load prediction history from database"""
        filter_symbol = self.filter_var.get()
        if filter_symbol == "All":
            filter_symbol = None
        
        predictions = self.db_manager.get_predictions(filter_symbol)
        
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Add predictions
        for pred in predictions:
            values = (
                pred['symbol'],
                f"${pred['predicted_price']:.2f}",
                f"${pred['actual_price']:.2f}" if pred['actual_price'] else "N/A",
                f"{pred['percentage_change']:+.2f}%",
                f"{pred['confidence']:.0%}",
                f"{pred['accuracy']:.1%}" if pred['accuracy'] else "N/A",
                datetime.fromisoformat(pred['timestamp']).strftime("%Y-%m-%d %H:%M")
            )
            self.history_tree.insert('', tk.END, values=values)
    
    def filter_predictions(self, event=None):
        """Filter predictions by symbol"""
        self.load_prediction_history()
    
    def update_notification_settings(self):
        """Update notification settings"""
        self.notification_manager.voice_enabled = self.voice_var.get()
        self.notification_manager.desktop_enabled = self.desktop_var.get()
    
    def load_settings(self):
        """Load settings from database"""
        # Load default settings
        self.filter_var.set("All")
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_monitoring()
        self.root.destroy()

# Main execution
if __name__ == "__main__":
    try:
        app = StockGuardGUI()
        app.run()
    except Exception as e:
        logger.error(f"Error starting StockGuard: {e}")
        print(f"Error: {e}")
        print("Please ensure all required libraries are installed:")
        print("pip install yfinance pandas numpy scikit-learn tensorflow tkinter requests pyttsx3 plyer")