from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import asyncio
import logging
from rag import fazer_consulta_juridica, qa_chain

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para o comando /start"""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    mensagem = """
👨‍⚖️ *Consultor Jurídico Bot*

Olá! Sou um assistente especializado em consultas jurídicas baseado em documentos legais.

*Como posso ajudá-lo:*
• Consultas sobre processos jurídicos
• Análise de fundamentações legais  
• Explicação de decisões judiciais
• Pesquisa em documentos jurídicos

*💡 Dica:* Seja específico em sua pergunta para obter respostas mais precisas.

Digite sua pergunta sobre qualquer aspecto jurídico!
    """
    await update.message.reply_text(mensagem, parse_mode=ParseMode.MARKDOWN)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para o comando /help"""
    help_text = """
*Comandos disponíveis:*
/start - Iniciar o bot
/help - Mostrar esta ajuda
/sobre - Informações sobre o bot
    """
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def sobre_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para o comando /sobre"""
    sobre_text = """
*🤖 Sobre este bot:*

*Funcionalidade:* Assistente jurídico baseado em RAG (Retrieval-Augmented Generation)
*Tecnologia:* IA especializada em análise de documentos jurídicos
*Base de conhecimento:* Documentos legais e jurisprudenciais

Desenvolvido para auxiliar em consultas jurídicas.
    """
    await update.message.reply_text(sobre_text, parse_mode=ParseMode.MARKDOWN)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para mensagens de texto dos usuários"""
    user_id = update.effective_user.id
    pergunta_usuario = update.message.text.strip()
    
    # Log da pergunta
    logger.info(f"Usuário {user_id} perguntou: {pergunta_usuario}")
    
    # Verificar se a pergunta não está vazia
    if not pergunta_usuario:
        await update.message.reply_text("Por favor, digite uma pergunta.")
        return
    
    try:
        # Indicar que o bot está digitando
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, 
            action=ChatAction.TYPING
        )
        
        # Processar a consulta (com timeout)
        resposta = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, fazer_consulta_juridica, qa_chain, pergunta_usuario
            ),
            timeout=30.0  # Timeout de 30 segundos
        )
        
        # Verificar se a resposta não está vazia
        if not resposta or len(resposta.strip()) == 0:
            resposta = "Não foi possível gerar uma resposta para sua pergunta. Tente reformular."
        
        # Enviar a resposta
        await update.message.reply_text(resposta)
        
        # Log de sucesso
        logger.info(f"Resposta enviada para usuário {user_id}")
        
    except asyncio.TimeoutError:
        error_msg = "⏰ A consulta está demorando mais que o esperado. Tente novamente com uma pergunta mais específica."
        await update.message.reply_text(error_msg)
        logger.warning(f"Timeout para usuário {user_id}")
        
    except Exception as e:
        error_msg = "❌ Ocorreu um erro ao processar sua pergunta. Tente novamente em alguns instantes."
        await update.message.reply_text(error_msg)
        logger.error(f"Erro para usuário {user_id}: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler global de erros"""
    logger.error(f"Erro não capturado: {context.error}", exc_info=context.error)

def main():
    """Função principal para iniciar o bot"""
    application = ApplicationBuilder().token("8113448825:AAGXnAl5NZx54RRc74it6EEI2GJ_ym6SXdg").build()
    
    # Adicionar handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("sobre", sobre_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Adicionar handler de erros
    application.add_error_handler(error_handler)
    
    # Iniciar o bot
    print("🤖 Bot do Consultor Jurídico iniciado...")
    application.run_polling()

if __name__ == "__main__":
    main()