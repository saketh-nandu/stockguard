import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Configure matplotlib for dark theme
plt.style.use('dark_background')

# List of 100 realistic stock symbols
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "INTC", "ORCL",
    "IBM", "CSCO", "BMY", "PFE", "LLY", "JNJ", "ABBV", "ZTS", "NVO", "MRK",
    "WBA", "CVS", "UNH", "ANTM", "CI", "HUM", "MOH", "REGN", "VRTX", "GILD",
    "BIIB", "ALNY", "IONS", "EXAS", "BNTX", "NVAX", "IMMR", "ADBE", "CRM", "SQ",
    "PYPL", "COIN", "MSTR", "OKTA", "DOCU", "BOX", "CRWD", "SNOW", "DB", "RHT",
    "NOW", "FIS", "FISV", "DXC", "HPQ", "DELL", "VMW", "UBER", "LYFT", "GM",
    "F", "TM", "HMC", "HYMTF", "MBTYY", "NSANY", "RACE", "VWAGY", "BMWYY", "LI",
    "XPEV", "NIO", "BABA", "NFLX", "DIS", "SHOP", "SPOT", "TWTR", "SNAP", "PINS",
    "ROKU", "ZOOM", "PTON", "PLTR", "RBLX", "RIVN", "LCID", "HOOD", "UPST", "AFRM",
    "SOFI", "OPEN", "WISH", "CLOV", "SPCE", "DKNG", "PENN", "MGM", "WYNN", "LVS"
]

class StockMarketSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Simulator Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg="#0d1117")
        
        # Set window icon and make it resizable
        self.root.minsize(1200, 800)
        
        # Configure custom style
        self.setup_styles()

        # Initialize data structures
        self.stock_data = {}
        self.portfolio = {}
        self.transactions = []
        self.alerts = []
        self.selected_symbol = None
        self.cash_balance = 10000.0  # Starting cash
        
        # Chart settings
        self.chart_type = tk.StringVar(value="candlestick")
        self.timeframe_var = tk.StringVar(value="1D")
        self.zoom_var = tk.BooleanVar()
        self.span_selector = None
        
        # Generate initial stock data
        self.generate_all_stock_data()
        
        # Create UI
        self.create_widgets()
        
        # Start background processes
        self.update_clock()
        self.simulate_price_changes()
        
    def setup_styles(self):
        """Configure custom ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Custom.TFrame', background='#0d1117')
        style.configure('Custom.TLabel', background='#0d1117', foreground='#c9d1d9')
        style.configure('Custom.TButton', background='#21262d', foreground='#c9d1d9')
        style.configure('Custom.TEntry', background='#21262d', foreground='#c9d1d9')
        style.configure('Treeview', background='#0d1117', foreground='#c9d1d9', fieldbackground='#0d1117')
        style.configure('Treeview.Heading', background='#21262d', foreground='#c9d1d9')
        
    def generate_all_stock_data(self):
        """Generate initial stock data for all symbols"""
        for symbol in STOCK_SYMBOLS:
            self.stock_data[symbol] = self.generate_stock_data(symbol)
    
    def generate_stock_data(self, symbol, days=30):
        """Generate realistic stock data with trends"""
        base_price = random.uniform(20, 500)
        data = []
        current_price = base_price
        
        # Generate data points for the specified number of days
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(days * 24):  # Hourly data
            timestamp = start_date + timedelta(hours=i)
            
            # Add some realistic volatility
            volatility = random.uniform(0.5, 2.0)
            change_percent = random.gauss(0, volatility) / 100
            
            open_price = current_price
            high_price = open_price * (1 + abs(change_percent) * random.uniform(0.5, 1.5))
            low_price = open_price * (1 - abs(change_percent) * random.uniform(0.5, 1.5))
            close_price = open_price * (1 + change_percent)
            
            # Ensure high is highest and low is lowest
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(10000, 1000000)
            })
            
            current_price = close_price
            
        return data
    
    def create_widgets(self):
        """Create the main UI layout"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel
        left_frame = ttk.Frame(main_paned, style='Custom.TFrame')
        main_paned.add(left_frame, weight=1)
        
        # Right panel
        right_frame = ttk.Frame(main_paned, style='Custom.TFrame')
        main_paned.add(right_frame, weight=3)
        
        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)
        
        # Bottom panel for portfolio and transactions
        bottom_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        bottom_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        self.create_bottom_panel()
        
    def create_left_panel(self, parent):
        """Create the left panel with stock list and search"""
        # Header
        header_frame = tk.Frame(parent, bg="#21262d", height=50)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="Stock Market", 
                              font=("Segoe UI", 14, "bold"), 
                              bg="#21262d", fg="#58a6ff")
        title_label.pack(pady=10)
        
        # Clock
        self.time_label = tk.Label(header_frame, text="", 
                                  font=("Segoe UI", 10), 
                                  bg="#21262d", fg="#8b949e")
        self.time_label.pack()
        
        # Cash balance
        cash_frame = tk.Frame(parent, bg="#0d1117")
        cash_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(cash_frame, text="Cash Balance:", 
                font=("Segoe UI", 10, "bold"), 
                bg="#0d1117", fg="#c9d1d9").pack()
        
        self.cash_label = tk.Label(cash_frame, text=f"${self.cash_balance:,.2f}", 
                                  font=("Segoe UI", 12, "bold"), 
                                  bg="#0d1117", fg="#3fb950")
        self.cash_label.pack()
        
        # Search
        search_frame = tk.Frame(parent, bg="#0d1117")
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(search_frame, text="Search Stocks:", 
                font=("Segoe UI", 10), 
                bg="#0d1117", fg="#c9d1d9").pack(anchor=tk.W)
        
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_stocks)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                               bg="#21262d", fg="#c9d1d9", insertbackground="#c9d1d9")
        search_entry.pack(fill=tk.X, pady=2)
        
        # Stock list
        list_frame = tk.Frame(parent, bg="#0d1117")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for listbox
        scrollbar = tk.Scrollbar(list_frame, bg="#21262d")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stock_listbox = tk.Listbox(list_frame, 
                                       bg="#0d1117", fg="#c9d1d9",
                                       selectbackground="#1f6feb",
                                       selectforeground="white",
                                       font=("Consolas", 10),
                                       yscrollcommand=scrollbar.set,
                                       bd=0, highlightthickness=0)
        self.stock_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.stock_listbox.yview)
        
        # Populate stock list
        self.populate_stock_list()
        
        # Bind selection event
        self.stock_listbox.bind('<<ListboxSelect>>', self.on_stock_select)
        
    def create_right_panel(self, parent):
        """Create the right panel with chart and controls"""
        # Top controls
        controls_frame = tk.Frame(parent, bg="#21262d", height=60)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        controls_frame.pack_propagate(False)
        
        # Stock info
        info_frame = tk.Frame(controls_frame, bg="#21262d")
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.symbol_label = tk.Label(info_frame, text="Select a stock", 
                                    font=("Segoe UI", 16, "bold"), 
                                    bg="#21262d", fg="#58a6ff")
        self.symbol_label.pack()
        
        self.price_label = tk.Label(info_frame, text="$0.00", 
                                   font=("Segoe UI", 14), 
                                   bg="#21262d", fg="#3fb950")
        self.price_label.pack()
        
        self.change_label = tk.Label(info_frame, text="", 
                                    font=("Segoe UI", 10), 
                                    bg="#21262d", fg="#8b949e")
        self.change_label.pack()
        
        # Chart controls
        chart_controls = tk.Frame(controls_frame, bg="#21262d")
        chart_controls.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Timeframe selection
        tk.Label(chart_controls, text="Timeframe:", 
                bg="#21262d", fg="#c9d1d9").grid(row=0, column=0, sticky=tk.W)
        
        timeframe_combo = ttk.Combobox(chart_controls, textvariable=self.timeframe_var,
                                      values=['1H', '1D', '1W', '1M'], 
                                      state='readonly', width=8)
        timeframe_combo.grid(row=0, column=1, padx=5)
        timeframe_combo.bind('<<ComboboxSelected>>', self.on_timeframe_change)
        
        # Chart type selection
        tk.Label(chart_controls, text="Chart:", 
                bg="#21262d", fg="#c9d1d9").grid(row=1, column=0, sticky=tk.W)
        
        chart_combo = ttk.Combobox(chart_controls, textvariable=self.chart_type,
                                  values=['candlestick', 'line'], 
                                  state='readonly', width=8)
        chart_combo.grid(row=1, column=1, padx=5)
        chart_combo.bind('<<ComboboxSelected>>', self.on_chart_type_change)
        
        # Action buttons
        action_frame = tk.Frame(parent, bg="#0d1117", height=50)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        action_frame.pack_propagate(False)
        
        button_frame = tk.Frame(action_frame, bg="#0d1117")
        button_frame.pack(expand=True)
        
        # Buy button
        buy_btn = tk.Button(button_frame, text="🛒 BUY", 
                           command=self.buy_stock,
                           bg="#238636", fg="white", 
                           font=("Segoe UI", 10, "bold"),
                           relief=tk.FLAT, padx=20, pady=5)
        buy_btn.pack(side=tk.LEFT, padx=5)
        
        # Sell button
        sell_btn = tk.Button(button_frame, text="💰 SELL", 
                            command=self.sell_stock,
                            bg="#da3633", fg="white", 
                            font=("Segoe UI", 10, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        sell_btn.pack(side=tk.LEFT, padx=5)
        
        # Alert button
        alert_btn = tk.Button(button_frame, text="🔔 ALERT", 
                             command=self.set_alert,
                             bg="#1f6feb", fg="white", 
                             font=("Segoe UI", 10, "bold"),
                             relief=tk.FLAT, padx=20, pady=5)
        alert_btn.pack(side=tk.LEFT, padx=5)
        
        # Export button
        export_btn = tk.Button(button_frame, text="📊 EXPORT", 
                              command=self.export_portfolio,
                              bg="#6f42c1", fg="white", 
                              font=("Segoe UI", 10, "bold"),
                              relief=tk.FLAT, padx=20, pady=5)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Chart area
        chart_frame = tk.Frame(parent, bg="#0d1117")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6), facecolor='#0d1117')
        self.ax.set_facecolor('#0d1117')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty chart
        self.plot_empty_chart()
        
    def create_bottom_panel(self):
        """Create bottom panel with portfolio and transactions"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Portfolio tab
        portfolio_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(portfolio_frame, text='📈 Portfolio')
        
        # Portfolio treeview
        port_columns = ('Symbol', 'Shares', 'Avg Cost', 'Current Price', 'Market Value', 'P&L', 'P&L %')
        self.portfolio_tree = ttk.Treeview(portfolio_frame, columns=port_columns, 
                                          show='headings', height=6)
        
        for col in port_columns:
            self.portfolio_tree.heading(col, text=col)
            self.portfolio_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Scrollbars for portfolio
        port_v_scroll = ttk.Scrollbar(portfolio_frame, orient=tk.VERTICAL, 
                                     command=self.portfolio_tree.yview)
        port_h_scroll = ttk.Scrollbar(portfolio_frame, orient=tk.HORIZONTAL, 
                                     command=self.portfolio_tree.xview)
        
        self.portfolio_tree.configure(yscrollcommand=port_v_scroll.set, 
                                     xscrollcommand=port_h_scroll.set)
        
        self.portfolio_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        port_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        port_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Transactions tab
        trans_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(trans_frame, text='📋 Transactions')
        
        # Transactions treeview
        trans_columns = ('Date', 'Type', 'Symbol', 'Quantity', 'Price', 'Total')
        self.transactions_tree = ttk.Treeview(trans_frame, columns=trans_columns, 
                                            show='headings', height=6)
        
        for col in trans_columns:
            self.transactions_tree.heading(col, text=col)
            self.transactions_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Scrollbars for transactions
        trans_v_scroll = ttk.Scrollbar(trans_frame, orient=tk.VERTICAL, 
                                      command=self.transactions_tree.yview)
        trans_h_scroll = ttk.Scrollbar(trans_frame, orient=tk.HORIZONTAL, 
                                      command=self.transactions_tree.xview)
        
        self.transactions_tree.configure(yscrollcommand=trans_v_scroll.set, 
                                        xscrollcommand=trans_h_scroll.set)
        
        self.transactions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trans_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        trans_h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Alerts tab
        alerts_frame = ttk.Frame(self.notebook, style='Custom.TFrame')
        self.notebook.add(alerts_frame, text='🔔 Alerts')
        
        # Alerts treeview
        alerts_columns = ('Symbol', 'Target Price', 'Type', 'Status')
        self.alerts_tree = ttk.Treeview(alerts_frame, columns=alerts_columns, 
                                       show='headings', height=6)
        
        for col in alerts_columns:
            self.alerts_tree.heading(col, text=col)
            self.alerts_tree.column(col, width=120, anchor=tk.CENTER)
        
        self.alerts_tree.pack(fill=tk.BOTH, expand=True)
        
    def populate_stock_list(self):
        """Populate the stock list with current prices"""
        self.stock_listbox.delete(0, tk.END)
        
        for symbol in sorted(STOCK_SYMBOLS):
            if symbol in self.stock_data and self.stock_data[symbol]:
                current_price = self.stock_data[symbol][-1]['close']
                display_text = f"{symbol:<6} ${current_price:>8.2f}"
                self.stock_listbox.insert(tk.END, display_text)
            else:
                self.stock_listbox.insert(tk.END, f"{symbol:<6} ${'N/A':>8}")
    
    def filter_stocks(self, *args):
        """Filter stocks based on search input"""
        search_term = self.search_var.get().upper()
        self.stock_listbox.delete(0, tk.END)
        
        for symbol in sorted(STOCK_SYMBOLS):
            if search_term in symbol:
                if symbol in self.stock_data and self.stock_data[symbol]:
                    current_price = self.stock_data[symbol][-1]['close']
                    display_text = f"{symbol:<6} ${current_price:>8.2f}"
                    self.stock_listbox.insert(tk.END, display_text)
                else:
                    self.stock_listbox.insert(tk.END, f"{symbol:<6} ${'N/A':>8}")
    
    def on_stock_select(self, event):
        """Handle stock selection"""
        selection = self.stock_listbox.curselection()
        if not selection:
            return
            
        selected_text = self.stock_listbox.get(selection[0])
        symbol = selected_text.split()[0]
        self.selected_symbol = symbol
        
        if symbol in self.stock_data and self.stock_data[symbol]:
            self.update_stock_display(symbol)
            self.plot_chart(symbol)
            self.check_alerts(symbol)
    
    def update_stock_display(self, symbol):
        """Update the stock information display"""
        if symbol not in self.stock_data or not self.stock_data[symbol]:
            return
            
        current_data = self.stock_data[symbol][-1]
        prev_data = self.stock_data[symbol][-2] if len(self.stock_data[symbol]) > 1 else current_data
        
        current_price = current_data['close']
        prev_price = prev_data['close']
        change = current_price - prev_price
        change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
        
        self.symbol_label.config(text=symbol)
        self.price_label.config(text=f"${current_price:.2f}")
        
        # Color code the change
        if change >= 0:
            change_color = "#3fb950"
            change_text = f"+${change:.2f} (+{change_percent:.2f}%)"
        else:
            change_color = "#f85149"
            change_text = f"-${abs(change):.2f} ({change_percent:.2f}%)"
            
        self.change_label.config(text=change_text, fg=change_color)
        self.price_label.config(fg=change_color)
    
    def plot_chart(self, symbol):
        """Plot the stock chart"""
        if symbol not in self.stock_data or not self.stock_data[symbol]:
            return
            
        data = self.stock_data[symbol]
        
        # Filter data based on timeframe
        timeframe = self.timeframe_var.get()
        if timeframe == '1H':
            data = data[-60:]  # Last 60 data points
        elif timeframe == '1D':
            data = data[-24:]  # Last 24 hours
        elif timeframe == '1W':
            data = data[-168:]  # Last week
        elif timeframe == '1M':
            data = data[-720:]  # Last month
        
        if not data:
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.ax.clear()
        
        if self.chart_type.get() == 'candlestick':
            self.plot_candlestick(df)
        else:
            self.plot_line(df)
            
        self.ax.set_title(f'{symbol} - {timeframe}', color='#c9d1d9', fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Time', color='#c9d1d9')
        self.ax.set_ylabel('Price ($)', color='#c9d1d9')
        self.ax.grid(True, alpha=0.3)
        self.ax.tick_params(colors='#c9d1d9')
        
        # Format x-axis
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(data)//10)))
        
        self.fig.autofmt_xdate()
        self.canvas.draw()
    
    def plot_candlestick(self, df):
        """Plot candlestick chart"""
        for i, row in df.iterrows():
            x = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Color
            color = '#3fb950' if close_price >= open_price else '#f85149'
            
            # High-low line
            self.ax.plot([x, x], [low_price, high_price], color=color, linewidth=1)
            
            # Body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            # Use bar to create the body
            self.ax.bar(x, body_height, bottom=body_bottom, 
                       color=color, alpha=0.8, width=timedelta(minutes=30))
    
    def plot_line(self, df):
        """Plot line chart"""
        self.ax.plot(df['timestamp'], df['close'], color='#58a6ff', linewidth=2)
        self.ax.fill_between(df['timestamp'], df['close'], alpha=0.3, color='#58a6ff')
    
    def plot_empty_chart(self):
        """Plot empty chart when no stock is selected"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Select a stock to view chart', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, color='#8b949e', fontsize=16)
        self.ax.set_facecolor('#0d1117')
        self.canvas.draw()
    
    def on_timeframe_change(self, event=None):
        """Handle timeframe change"""
        if self.selected_symbol:
            self.plot_chart(self.selected_symbol)
    
    def on_chart_type_change(self, event=None):
        """Handle chart type change"""
        if self.selected_symbol:
            self.plot_chart(self.selected_symbol)
    
    def buy_stock(self):
        """Buy stock functionality"""
        if not self.selected_symbol:
            messagebox.showwarning("No Selection", "Please select a stock first.")
            return
            
        symbol = self.selected_symbol
        if symbol not in self.stock_data or not self.stock_data[symbol]:
            messagebox.showerror("Error", "No price data available for this stock.")
            return
            
        current_price = self.stock_data[symbol][-1]['close']
        
        # Ask for quantity
        quantity = simpledialog.askinteger("Buy Stock", 
                                          f"Enter number of shares to buy:\n"
                                          f"Current price: ${current_price:.2f}\n"
                                          f"Available cash: ${self.cash_balance:.2f}",
                                          minvalue=1, maxvalue=int(self.cash_balance // current_price))
        
        if quantity:
            total_cost = quantity * current_price
            
            if total_cost <= self.cash_balance:
                # Confirm purchase
                if messagebox.askyesno("Confirm Purchase", 
                                     f"Buy {quantity} shares of {symbol} at ${current_price:.2f}?\n"
                                     f"Total cost: ${total_cost:.2f}"):
                    
                    # Update portfolio
                    if symbol in self.portfolio:
                        old_qty = self.portfolio[symbol]['quantity']
                        old_avg_cost = self.portfolio[symbol]['avg_cost']
                        new_avg_cost = ((old_qty * old_avg_cost) + total_cost) / (old_qty + quantity)
                        self.portfolio[symbol]['quantity'] += quantity
                        self.portfolio[symbol]['avg_cost'] = new_avg_cost
                    else:
                        self.portfolio[symbol] = {'quantity': quantity, 'avg_cost': current_price}
                    
                    # Update cash balance
                    self.cash_balance -= total_cost
                    self.cash_label.config(text=f"${self.cash_balance:,.2f}")
                    
                    # Record transaction
                    self.transactions.append({
                        'date': datetime.now(),
                        'type': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': current_price,
                        'total': total_cost
                    })
                    
                    # Update displays
                    self.update_portfolio_display()
                    self.update_transactions_display()
                    
                    messagebox.showinfo("Success", f"Successfully bought {quantity} shares of {symbol}")
            else:
                messagebox.showerror("Insufficient Funds", 
                                   f"You need ${total_cost:.2f} but only have ${self.cash_balance:.2f}")
    
    def sell_stock(self):
        """Sell stock functionality"""
        if not self.selected_symbol:
            messagebox.showwarning("No Selection", "Please select a stock first.")
            return
            
        symbol = self.selected_symbol
        if symbol not in self.portfolio or self.portfolio[symbol]['quantity'] <= 0:
            messagebox.showwarning("No Holdings", f"You don't own any shares of {symbol}.")
            return
            
        if symbol not in self.stock_data or not self.stock_data[symbol]:
            messagebox.showerror("Error", "No price data available for this stock.")
            return
            
        current_price = self.stock_data[symbol][-1]['close']
        owned_shares = self.portfolio[symbol]['quantity']
        
        # Ask for quantity to sell
        quantity = simpledialog.askinteger("Sell Stock", 
                                          f"Enter number of shares to sell:\n"
                                          f"Current price: ${current_price:.2f}\n"
                                          f"Owned shares: {owned_shares}",
                                          minvalue=1, maxvalue=owned_shares)
        
        if quantity:
            total_value = quantity * current_price
            
            # Confirm sale
            if messagebox.askyesno("Confirm Sale", 
                                 f"Sell {quantity} shares of {symbol} at ${current_price:.2f}?\n"
                                 f"Total value: ${total_value:.2f}"):
                
                # Update portfolio
                self.portfolio[symbol]['quantity'] -= quantity
                if self.portfolio[symbol]['quantity'] == 0:
                    del self.portfolio[symbol]
                
                # Update cash balance
                self.cash_balance += total_value
                self.cash_label.config(text=f"${self.cash_balance:,.2f}")
                
                # Record transaction
                self.transactions.append({
                    'date': datetime.now(),
                    'type': 'SELL',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': current_price,
                    'total': total_value
                })
                
                # Update displays
                self.update_portfolio_display()
                self.update_transactions_display()
                
                messagebox.showinfo("Success", f"Successfully sold {quantity} shares of {symbol}")
    
    def set_alert(self):
        """Set price alert functionality"""
        if not self.selected_symbol:
            messagebox.showwarning("No Selection", "Please select a stock first.")
            return
            
        symbol = self.selected_symbol
        if symbol not in self.stock_data or not self.stock_data[symbol]:
            messagebox.showerror("Error", "No price data available for this stock.")
            return
            
        current_price = self.stock_data[symbol][-1]['close']
        
        # Ask for target price
        target_price = simpledialog.askfloat("Set Price Alert", 
                                           f"Current price of {symbol}: ${current_price:.2f}\n"
                                           f"Enter target price for alert:",
                                           minvalue=0.01)
        
        if target_price:
            alert_type = "Above" if target_price > current_price else "Below"
            
            # Add alert
            self.alerts.append({
                'symbol': symbol,
                'target_price': target_price,
                'type': alert_type,
                'status': 'Active',
                'created_date': datetime.now()
            })
            
            self.update_alerts_display()
            messagebox.showinfo("Alert Set", 
                              f"Alert set for {symbol} when price goes {alert_type.lower()} ${target_price:.2f}")
    
    def export_portfolio(self):
        """Export portfolio to CSV"""
        if not self.portfolio:
            messagebox.showwarning("Empty Portfolio", "Your portfolio is empty.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Portfolio"
        )
        
        if filename:
            try:
                # Prepare data
                data = []
                total_value = 0
                
                for symbol, holding in self.portfolio.items():
                    if symbol in self.stock_data and self.stock_data[symbol]:
                        current_price = self.stock_data[symbol][-1]['close']
                        quantity = holding['quantity']
                        avg_cost = holding['avg_cost']
                        market_value = quantity * current_price
                        total_cost = quantity * avg_cost
                        pnl = market_value - total_cost
                        pnl_percent = (pnl / total_cost) * 100 if total_cost > 0 else 0
                        
                        data.append([
                            symbol, quantity, f"{avg_cost:.2f}", f"{current_price:.2f}",
                            f"{market_value:.2f}", f"{pnl:.2f}", f"{pnl_percent:.2f}%"
                        ])
                        total_value += market_value
                
                # Create DataFrame and save
                df = pd.DataFrame(data, columns=[
                    'Symbol', 'Shares', 'Avg Cost', 'Current Price', 
                    'Market Value', 'P&L', 'P&L %'
                ])
                
                df.to_csv(filename, index=False)
                
                messagebox.showinfo("Export Successful", 
                                  f"Portfolio exported to {os.path.basename(filename)}\n"
                                  f"Total Portfolio Value: ${total_value:.2f}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export portfolio: {str(e)}")
    
    def check_alerts(self, symbol):
        """Check and trigger alerts"""
        if symbol not in self.stock_data or not self.stock_data[symbol]:
            return
            
        current_price = self.stock_data[symbol][-1]['close']
        
        # Check active alerts for this symbol
        triggered_alerts = []
        
        for alert in self.alerts:
            if (alert['symbol'] == symbol and alert['status'] == 'Active'):
                target_price = alert['target_price']
                alert_type = alert['type']
                
                if ((alert_type == "Above" and current_price >= target_price) or
                    (alert_type == "Below" and current_price <= target_price)):
                    
                    alert['status'] = 'Triggered'
                    triggered_alerts.append(alert)
                    
                    # Show alert notification
                    messagebox.showinfo("Price Alert Triggered!", 
                                      f"{symbol} has reached ${current_price:.2f}\n"
                                      f"Target was ${target_price:.2f} ({alert_type})")
        
        if triggered_alerts:
            self.update_alerts_display()
    
    def update_portfolio_display(self):
        """Update the portfolio display"""
        # Clear existing items
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        # Add portfolio items
        total_value = 0
        total_cost = 0
        
        for symbol, holding in self.portfolio.items():
            if symbol in self.stock_data and self.stock_data[symbol]:
                current_price = self.stock_data[symbol][-1]['close']
                quantity = holding['quantity']
                avg_cost = holding['avg_cost']
                market_value = quantity * current_price
                cost_basis = quantity * avg_cost
                pnl = market_value - cost_basis
                pnl_percent = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Color code P&L
                pnl_color = '#3fb950' if pnl >= 0 else '#f85149'
                
                self.portfolio_tree.insert('', 'end', values=(
                    symbol,
                    f"{quantity:,}",
                    f"${avg_cost:.2f}",
                    f"${current_price:.2f}",
                    f"${market_value:,.2f}",
                    f"${pnl:,.2f}",
                    f"{pnl_percent:.2f}%"
                ))
                
                total_value += market_value
                total_cost += cost_basis
        
        # Add total row
        if self.portfolio:
            total_pnl = total_value - total_cost
            total_pnl_percent = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            self.portfolio_tree.insert('', 'end', values=(
                "TOTAL",
                "",
                "",
                "",
                f"${total_value:,.2f}",
                f"${total_pnl:,.2f}",
                f"{total_pnl_percent:.2f}%"
            ))
    
    def update_transactions_display(self):
        """Update the transactions display"""
        # Clear existing items
        for item in self.transactions_tree.get_children():
            self.transactions_tree.delete(item)
        
        # Add recent transactions (last 50)
        recent_transactions = self.transactions[-50:] if len(self.transactions) > 50 else self.transactions
        
        for transaction in reversed(recent_transactions):
            self.transactions_tree.insert('', 'end', values=(
                transaction['date'].strftime('%Y-%m-%d %H:%M'),
                transaction['type'],
                transaction['symbol'],
                f"{transaction['quantity']:,}",
                f"${transaction['price']:.2f}",
                f"${transaction['total']:,.2f}"
            ))
    
    def update_alerts_display(self):
        """Update the alerts display"""
        # Clear existing items
        for item in self.alerts_tree.get_children():
            self.alerts_tree.delete(item)
        
        # Add alerts
        for alert in self.alerts:
            self.alerts_tree.insert('', 'end', values=(
                alert['symbol'],
                f"${alert['target_price']:.2f}",
                alert['type'],
                alert['status']
            ))
    
    def update_clock(self):
        """Update the clock display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_clock)
    
    def simulate_price_changes(self):
        """Simulate real-time price changes"""
        for symbol in STOCK_SYMBOLS:
            if symbol in self.stock_data and self.stock_data[symbol]:
                last_data = self.stock_data[symbol][-1]
                last_price = last_data['close']
                
                # Generate new price with some volatility
                change_percent = random.gauss(0, 0.5) / 100  # 0.5% std deviation
                new_price = last_price * (1 + change_percent)
                
                # Create new data point
                new_data = {
                    'timestamp': datetime.now(),
                    'open': last_price,
                    'high': max(last_price, new_price) * (1 + random.uniform(0, 0.01)),
                    'low': min(last_price, new_price) * (1 - random.uniform(0, 0.01)),
                    'close': new_price,
                    'volume': random.randint(10000, 1000000)
                }
                
                # Add to data and keep only recent points
                self.stock_data[symbol].append(new_data)
                if len(self.stock_data[symbol]) > 1000:  # Keep last 1000 points
                    self.stock_data[symbol] = self.stock_data[symbol][-1000:]
        
        # Update displays
        if self.selected_symbol:
            self.update_stock_display(self.selected_symbol)
            self.plot_chart(self.selected_symbol)
            self.check_alerts(self.selected_symbol)
        
        # Update stock list prices
        self.populate_stock_list()
        
        # Update portfolio values
        if self.portfolio:
            self.update_portfolio_display()
        
        # Schedule next update
        self.root.after(3000, self.simulate_price_changes)  # Update every 3 seconds


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = StockMarketSimulator(root)
    
    # Set minimum window size
    root.minsize(1200, 800)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()