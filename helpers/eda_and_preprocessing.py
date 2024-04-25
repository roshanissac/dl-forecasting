import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

def feature_engineering_selection(data):

    # Convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    #Creating Sales/Customer Feature
    data['Sales/Customer'] = data['Sales']/data['Customers']
    
   
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)
    
    
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] =data.Date.dt.isocalendar().week

    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) +         (data.Month - data.CompetitionOpenSinceMonth)
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) +         (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1

    data = data.drop(['PromoInterval','monthStr'], axis=1)

    print("Finished Feature Engineering/Selection....New features created Sales/Customer,Year,Month,Day,DayOfWeek,WeekOfYear,IsPromoMonth,CompetitionOpen,PromoOpen")

    return data


def exploratory_data_analysis(data):
  
 
  # The dataset contains the values Sales which are 0 due to School or State Holiday.              
  # Plot the train data having sales greater then 0 for Store1   
  # It can be seen the sales jumps during end of December each year
  
  print("Generating and saving plot for sales of store1....")
  # Sales for Store 1
  strain = data[data.Sales>0]
  strain.loc[strain['Store']==1 ,['Date','Sales']].plot(x='Date',y='Sales',title='Store1',figsize=(16,4))
  plt.savefig("charts/EDA/sales_store1.png")
  plt.close()


  print("Generating and saving heatmap....")
  plt.figure(figsize = (20, 10))
  heat_map=sns.heatmap(data.corr(), annot = True, vmin = -1, vmax = 1, fmt = '.3f')
  fig = heat_map.get_figure()
  fig.savefig("charts/EDA/heatmap.png") 
  plt.close()

  print("Generating and saving monthly sales trend....")
  # Grouping by year and month, and summing up the sales for each group
  monthly_sales = data.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
  # Plotting the monthly sales trend
  plt.figure(figsize=(12, 6))
  plt.plot(monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str), monthly_sales['Sales'], marker='o')
  plt.title('Monthly Sales Over Time')
  plt.xlabel('Year-Month')
  plt.ylabel('Sales')
  plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
  plt.grid(True)
  plt.tight_layout()
  plt.savefig("charts/EDA/monthly_sales.png")
  plt.close()
  
  print("Generating and saving distribution of target column sales....")
  # Distribution of target column named 'sales'
  distribution_of_sales=sns.histplot(data=data, x='Sales')
  fig = distribution_of_sales.get_figure()
  fig.savefig("charts/EDA/distribution_of_sales.png") 
  plt.close()

  print("Generating and saving scatter plot of sales vs customers....")
  # Distribution of target column named 'sales'
  plt.figure(figsize=(20, 10))
  # Sales vs Customers Scatter Plot
  scatter_plot_sales_vs_customers=sns.scatterplot(x=data.Sales, y=data.Customers)
  plt.title("Sales vs Customers")
  fig = scatter_plot_sales_vs_customers.get_figure()
  fig.savefig("charts/EDA/scatter_plot_sales_vs_customers.png") 
  plt.close()

  print("Generating and saving bar plot of sales vs dayofweek....")
  bar_plot_sales_vs_dayofweek=sns.barplot(x=data.DayOfWeek, y=data.Sales)
  plt.title("Sales vs DayOfWeek")
  fig = bar_plot_sales_vs_dayofweek.get_figure()
  fig.savefig("charts/EDA/bar_plot_sales_vs_dayofweek.png")
  plt.close() 
  
  print("Generating and saving Sum of Sales per StoreType....")
  fig, ax = plt.subplots(1, 1, figsize = (23, 8))
  sns.lineplot(x='Date',y='Sales',hue='StoreType',data=(data.groupby(['Date', 'StoreType']).Sales.sum().to_frame()))
  ax.set_title("Sum of Sales per StoreType ")
  plt.savefig("charts/EDA/sum_of_sales_per_storetype.png")
  plt.close() 

  print("Generating and saving Sum of Sales per Assortment....")
  fig, ax = plt.subplots(1, 1, figsize = (23, 8))
  sns.lineplot(x='Date',y='Sales',hue='Assortment',data=(data.groupby(['Date', 'Assortment']).Sales.sum().to_frame()))
  ax.set_title("Sum of Sales per Assortment ")
  plt.savefig("charts/EDA/sum_of_sales_per_assortment.png")
  plt.close() 

  print("Generating and saving density plot for Sales per StoreType....")
  fig, ax = plt.subplots(1,1, figsize=(20, 5))
  density_plot_sales_per_store_type=sns.kdeplot(data=data, x='Sales', hue = 'StoreType', fill=True, alpha = 0.2, ax = ax)
  ax.set_title('Density plot for Sales per StoreType', fontweight = 'bold')
  fig = density_plot_sales_per_store_type.get_figure()
  fig.savefig("charts/EDA/density_plot_sales_per_store_type.png")
  plt.close()  

  print("Generating and saving density plot for Sales per Assortment....")
  fig, ax = plt.subplots(1,1, figsize=(20, 5))
  density_plot_sales_per_assortment=sns.kdeplot(data=data, x='Sales', hue = 'Assortment', fill=True, alpha = 0.2, ax = ax)
  ax.set_title('Density plot for Sales per Assortment', fontweight = 'bold')
  fig = density_plot_sales_per_assortment.get_figure()
  fig.savefig("charts/EDA/density_plot_sales_per_assortment.png") 
  plt.close() 

  print("Generating and saving box plot for grouped by SalesType....")
  fig, ax = plt.subplots(1, 1, figsize = (23, 5))
  data.set_index("Date")[["StoreType", "Sales"]].boxplot(by="StoreType", ax=ax)
  plt.savefig("charts/EDA/box_plot_groupedby_salestype.png")
  plt.close() 

  print("Generating and saving box plot for grouped by Year....")
  fig, ax = plt.subplots(1, 1, figsize = (23, 5))
  df_box = data.set_index("Date")
  df_box['year'] = df_box.index.to_period('Y')
  df_box['month'] = df_box.index.to_period('M')
  df_box[["year", "Sales"]].boxplot(by="year", ax=ax)
  plt.xticks(rotation=45);
  plt.savefig("charts/EDA/box_plot_groupedby_year.png")
  plt.close() 

  print("Generating and saving density plot for sales per weekday....")
  fig, ax = plt.subplots(1,1, figsize=(20, 5))
  density_plot_sales_per_weekday=sns.kdeplot(data=data, x='Sales', hue =data.set_index("Date").index.day_name(), fill=True, alpha = 0.1, ax = ax)
  ax.set_title('Density plot for Sales per weekday', fontweight = 'bold')
  fig = density_plot_sales_per_weekday.get_figure()
  fig.savefig("charts/EDA/density_plot_sales_per_weekday.png") 
  plt.close() 

  print("Generating and saving density plot for sales per month....")
  fig, ax = plt.subplots(1,1, figsize=(20, 5))
  density_plot_sales_per_month=sns.kdeplot(data=data, x='Sales', hue =data.set_index("Date").index.month, fill=True, alpha = 0.2, ax = ax)
  ax.set_title('Density plot for Sales per Month', fontweight = 'bold')
  fig = density_plot_sales_per_month.get_figure()
  fig.savefig("charts/EDA/density_plot_sales_per_month.png") 
  plt.close() 

  print("Generating and saving  plot for sales for store 1 with school holiday....")
  fig, ax = plt.subplots(1, 1, figsize = (23, 5), sharex=True)
  df_plot = data.groupby(['Date', 'Store', 'StateHoliday', 'SchoolHoliday']).Sales.sum().to_frame()
  df_plot = pd.DataFrame(df_plot.to_records()).set_index("Date")
  store=1
  zoom=slice("2014", "2014")
  plot_sales_for_store1_with_school_holidays=sns.lineplot(x='Date',y='Sales',hue='Store', data=df_plot.loc[df_plot["Store"]==store].loc[zoom], ax=ax, marker="o")
  ax.vlines(x=df_plot.loc[(df_plot["SchoolHoliday"]) & (df_plot["Store"]==store)].loc[zoom].index, ymin=1250, ymax=3500, color='gold', alpha=0.6, label="SchoolHoliday")
  plt.legend()
  ax.set_title("Sum of Sales in Store 1")
  fig = plot_sales_for_store1_with_school_holidays.get_figure()
  fig.savefig("charts/EDA/plot_sales_for_store1_with_school_holidays.png") 
  plt.close() 

  print("Generating and saving  plot for sales for store 1 with State holiday....")
  fig, ax = plt.subplots(1, 1, figsize = (23, 5), sharex=True)
  df_plot = data.groupby(['Date', 'Store', 'StateHoliday', 'SchoolHoliday']).Sales.sum().to_frame()
  df_plot = pd.DataFrame(df_plot.to_records()).set_index("Date")
  store=1
  zoom=slice("2014", "2014")
  plot_sales_for_store1_with_state_holidays=sns.lineplot(x='Date',y='Sales',hue='Store', data=df_plot.loc[df_plot["Store"]==store].loc[zoom], ax=ax, marker="o")
  ax.vlines(x=df_plot.loc[(df_plot["StateHoliday"]) & (df_plot["Store"]==store)].loc[zoom].index, ymin=1250, ymax=3500, color='gold', alpha=0.6, label="StateHoliday")
  plt.legend()
  ax.set_title("Sum of Sales in Store 1")
  fig = plot_sales_for_store1_with_state_holidays.get_figure()
  fig.savefig("charts/EDA/plot_sales_for_store1_with_state_holidays.png") 
  plt.close() 

  print("EDA Completed and Generated 15 graphs.")
 
  return data

def preprocess_data(train_path,store_path,columns_to_select=[]):


    print("Preprocessing Data with EDA.....")

    # Start time
    start_time = time.time()


    raw_train_df = pd.read_csv(train_path,low_memory=False)
    raw_stores_df = pd.read_csv(store_path,low_memory=False)

    # Merge store df and train df for better prediction
    train_merged_df = raw_train_df.merge(raw_stores_df, how='left', on='Store')
 

    """
    When the store is closed, the sale is zero. So rows with sales 0 doesn't make sense.
    But it's more efficient to remove rows with store 0. 
    """
    train_merged_df = train_merged_df[ train_merged_df.Open == 1 ].copy()

    print("Checking any columns with NaN values...")
    print("\n")
    print(train_merged_df.isnull().sum())

    
    #Fill the null values of Train dataset
    # fillna in store with 0 has better result than median()
    train_merged_df.fillna(0, inplace=True)

    print("Columns with NaN Values after replacing with 0...")
    print("\n")
    print(train_merged_df.isnull().sum())

    print("Starting Feature Engineering/Selection....")
    train_merged_df=feature_engineering_selection(train_merged_df)

    print("Starting Exploratory Data Analysis(EDA) and Generating Graphs.....")
    train_merged_df=exploratory_data_analysis(train_merged_df)

    train_merged_df.to_csv("datasets/preprocessed/preprocessed_training_data.csv")

    if columns_to_select:
      train_merged_df=train_merged_df[columns_to_select]

    train_data=train_merged_df.reset_index()

    train = (train_data.merge((train_data[['Date']].drop_duplicates(ignore_index=True).rename_axis('time_idx')).reset_index(), on = ['Date'])).drop(["Date","index"], axis=1)


    # End time
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time

    print(f"Preprocessing Data with EDA completed in {execution_time} seconds")

    return train

