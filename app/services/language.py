"""
Language support utilities including mappings and translation.
"""

# Language code to full name mapping for LLM instructions
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'pt': 'Portuguese',
    'th': 'Thai'
}

# Service acknowledgment messages in different languages
SERVICE_MESSAGES = {
    'en': "I've notified the cabin crew about your **{service_type}** request. A flight attendant will assist you at seat **{seat_number}** shortly.\n\n> ✅ Your request has been sent to the crew dashboard.",
    'es': "He notificado a la tripulación de cabina sobre su solicitud de **{service_type}**. Un auxiliar de vuelo le asistirá en el asiento **{seat_number}** en breve.\n\n> ✅ Su solicitud ha sido enviada al panel de la tripulación.",
    'fr': "J'ai informé l'équipage de cabine de votre demande de **{service_type}**. Un membre du personnel de bord vous assistera au siège **{seat_number}** sous peu.\n\n> ✅ Votre demande a été envoyée au tableau de bord de l'équipage.",
    'de': "Ich habe die Kabinenbesatzung über Ihre **{service_type}**-Anfrage informiert. Ein Flugbegleiter wird Ihnen am Sitz **{seat_number}** in Kürze behilflich sein.\n\n> ✅ Ihre Anfrage wurde an das Crew-Dashboard gesendet.",
    'hi': "मैंने आपके **{service_type}** अनुरोध के बारे में केबिन क्रू को सूचित कर दिया है। एक फ्लाइट अटेंडेंट जल्द ही सीट **{seat_number}** पर आपकी सहायता करेगा।\n\n> ✅ आपका अनुरोध क्रू डैशबोर्ड पर भेज दिया गया है।",
    'pt': "Notifiquei a tripulação de cabine sobre seu pedido de **{service_type}**. Um comissário de bordo irá ajudá-lo no assento **{seat_number}** em breve.\n\n> ✅ Seu pedido foi enviado ao painel da tripulação.",
    'th': "ฉันได้แจ้งลูกเรือเกี่ยวกับคำขอ **{service_type}** ของคุณแล้ว พนักงานต้อนรับจะมาช่วยเหลือคุณที่ที่นั่ง **{seat_number}** ในไม่ช้า\n\n> ✅ คำขอของคุณถูกส่งไปยังแดชบอร์ดลูกเรือแล้ว"
}


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    return LANGUAGE_NAMES.get(code, 'English')


def get_service_message(language: str, service_type: str, seat_number: str) -> str:
    """Get localized service acknowledgment message."""
    template = SERVICE_MESSAGES.get(language, SERVICE_MESSAGES['en'])
    return template.format(service_type=service_type, seat_number=seat_number)


async def translate_to_english(text: str, source_language: str, llm_interface) -> str:
    """Translate non-English queries to English for ChromaDB search."""
    if source_language == 'en':
        return text
    
    language_name = LANGUAGE_NAMES.get(source_language, 'Unknown')
    
    translation_prompt = f"""Translate the following {language_name} text to English. 
Only return the English translation, nothing else. Do not add any explanations or notes.

Text: {text}

English translation:"""
    
    try:
        translated = await llm_interface.generate_response(translation_prompt, temperature=0.1)
        # Clean up the translation - remove any quotes or extra formatting
        translated = translated.strip().strip('"').strip("'")
        print(f"🌐 Translated from {language_name}: '{text[:50]}...' → '{translated[:50]}...'")
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to original if translation fails
