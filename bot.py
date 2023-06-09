import telebot
from telebot import types

token = '6047459217:AAE_SweHllS_mbVSio7tz2wOdmghUpG3E7M'
bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет_05')
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("IT")
    item2 = types.KeyboardButton("Добывающая промышленность")
    item3 = types.KeyboardButton("Производственная промышленность")
    item4 = types.KeyboardButton("Финансовый сектор")
    item5 = types.KeyboardButton("Продукты питания")
    markup.add(item1)
    markup.add(item2)
    markup.add(item3)
    markup.add(item4)
    markup.add(item5)
    bot.send_message(message.chat.id, 'Выберите интересующую отрасль экономики', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def message_reply(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

    if message.text == "IT":
        tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'ADBE', 'INTC', 'CSCO']
        for ticker in tickers:
            item1 = types.KeyboardButton(ticker)
            markup.add(item1)
    if message.text == "Добывающая промышленность":
        tickers = ['XOM', 'CVX', 'RDS.A', 'RDS.B', 'PTR', 'BHP', 'BP', 'TOT', 'SNP', 'ENB']
        for ticker in tickers:
            item2 = types.KeyboardButton(ticker)
            markup.add(item2)
    if message.text == "Производственная промышленность":
        tickers = ['GE', 'HON', 'BA', 'LMT', 'MMM', 'CAT', 'UNP', 'UPS', 'GD', 'DE']
        for ticker in tickers:
            item3 = types.KeyboardButton(ticker)
            markup.add(item3)
    if message.text == "Финансовый сектор":
        tickers = ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'USB', 'BK', 'PNC', 'TD']
        for ticker in tickers:
            item4 = types.KeyboardButton(ticker)
            markup.add(item4)
    if message.text == "Продукты питания":
        tickers = ['KO', 'PEP', 'PG', 'Nestle', 'KHC', 'MDLZ', 'UL', 'GIS', 'K', 'KMB']
        for ticker in tickers:
            item5 = types.KeyboardButton(ticker)
            markup.add(item5)

    bot.send_message(message.chat.id, 'Выберите тикер', reply_markup=markup)


@bot.message_handler(content_types=['text'])
def analyze_data(message):
    ticker = message.test()

    bot.send_message(message.chat.id, f'Тикер выбран, спасибо! Ваш тикер: {ticker}')


bot.infinity_polling()
