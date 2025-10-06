from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from rag import fazer_consulta_juridica, qa_chain

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    mensagem = """
👨‍⚖️ *Consultor Jurídico Bot*

Olá! Sou um assistente especializado em consultas jurídicas baseado em documentos legais.

*Como posso ajudá-lo:*
• Consultas sobre processos jurídicos
• Análise de fundamentações legais
• Explicação de decisões judiciais
• Pesquisa em documentos jurídicos

Digite sua pergunta sobre qualquer aspecto jurídico!
    """
    await update.message.reply_text(mensagem, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pergunta_usuario = update.message.text
    resposta = fazer_consulta_juridica(qa_chain, pergunta_usuario)
    await update.message.reply_text(resposta)

def main():
    application = ApplicationBuilder().token("8113448825:AAGXnAl5NZx54RRc74it6EEI2GJ_ym6SXdg").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    main()