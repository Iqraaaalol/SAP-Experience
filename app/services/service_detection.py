"""
Multilingual service request detection for in-flight services.
Supports: English, Spanish, French, German, Portuguese, Hindi, Thai
"""
import re
from typing import Dict, Optional


# Multilingual travel indicators - skip service detection for travel queries
TRAVEL_INDICATORS = [
    # English
    r'(in|at|around|near)\s+(the\s+)?\w+',
    r'what.*(can|should|to)\s+(i\s+)?(eat|drink|try|visit|see|do)',
    r'(best|good|popular|famous|local|traditional)\s+(food|restaurant|cuisine|dish|drink)',
    r'(recommend|suggestion|tip|advice)',
    r'(where|when|how)\s+(to|can|should)',
    r'(tell me about|information|info)\s+',
    # Spanish
    r'(en|cerca de|alrededor de)\s+\w+',
    r'(qué|cuál|dónde|cómo|cuándo).*(comer|beber|visitar|ver|hacer)',
    r'(mejor|bueno|popular|famoso|local|tradicional)\s+(comida|restaurante|cocina|plato)',
    r'(recomendar|sugerencia|consejo)',
    # French
    r'(à|en|près de|autour de)\s+\w+',
    r'(que|quoi|où|comment|quand).*(manger|boire|visiter|voir|faire)',
    r'(meilleur|bon|populaire|célèbre|local|traditionnel)\s+(nourriture|restaurant|cuisine|plat)',
    # German
    r'(in|bei|um|nahe)\s+\w+',
    r'(was|wo|wie|wann).*(essen|trinken|besuchen|sehen|machen)',
    r'(beste|gut|beliebt|berühmt|lokal|traditionell)\s+(essen|restaurant|küche|gericht)',
    # Portuguese
    r'(em|perto de|ao redor de)\s+\w+',
    r'(o que|qual|onde|como|quando).*(comer|beber|visitar|ver|fazer)',
    r'(melhor|bom|popular|famoso|local|tradicional)\s+(comida|restaurante|cozinha|prato)',
    # Hindi (transliterated)
    r'(में|पर|के पास)\s+',
    r'(क्या|कहाँ|कैसे|कब).*(खाना|पीना|देखना|करना)',
    # Thai (transliterated common phrases)
    r'(ที่|ใน|ใกล้)',
    r'(อะไร|ที่ไหน|อย่างไร).*(กิน|ดื่ม|เที่ยว|ดู)',
]

# Multilingual service patterns
SERVICE_PATTERNS = {
    'beverage': {
        'patterns': [
            # English
            r'(can\s+i\s+(get|have)|i\'?d?\s+like|bring\s+me|i\s+need|i\s+want|please\s+get)\s*(a\s+|some\s+)?(water|drink|beverage|coffee|tea|juice|soda|wine|beer)',
            r'^(water|coffee|tea|juice|soda|wine|beer)(\s+please)?\.?$',
            r'i\'?m\s+(very\s+)?thirsty',
            # Spanish
            r'(puedo\s+tener|me\s+trae|quiero|necesito|quisiera)\s*(un|una|algo\s+de)?\s*(agua|bebida|café|té|jugo|refresco|vino|cerveza)',
            r'tengo\s+(mucha\s+)?sed',
            # French
            r'(puis-je\s+avoir|je\s+voudrais|apportez-moi|j\'?ai\s+besoin)\s*(d\'?|de\s+l\'?)?(eau|boisson|café|thé|jus|soda|vin|bière)',
            r'j\'?ai\s+(très\s+)?soif',
            # German
            r'(kann\s+ich|ich\s+möchte|bringen\s+sie\s+mir|ich\s+brauche|ich\s+hätte\s+gern)\s*(ein|eine|etwas)?\s*(wasser|getränk|kaffee|tee|saft|cola|wein|bier)',
            r'ich\s+habe\s+(großen\s+)?durst',
            # Portuguese
            r'(posso\s+ter|me\s+traz|quero|preciso|gostaria)\s*(um|uma|algo)?\s*(água|bebida|café|chá|suco|refrigerante|vinho|cerveja)',
            r'estou\s+com\s+(muita\s+)?sede',
            # Hindi
            r'(पानी|चाय|कॉफी|जूस|पेय)\s*(चाहिए|दीजिए|लाइए)',
            r'(मुझे\s+)?प्यास\s+लगी',
            # Thai
            r'(ขอ|ต้องการ|อยากได้)\s*(น้ำ|กาแฟ|ชา|น้ำผลไม้|เครื่องดื่ม)',
            r'หิวน้ำ|กระหายน้ำ',
        ]
    },
    'food': {
        'patterns': [
            # English
            r'(can\s+i\s+(get|have)|i\'?d?\s+like|bring\s+me|i\s+need|i\s+want|please\s+get)\s*(a\s+|some\s+)?(food|meal|snack|sandwich)',
            r'^(meal|snack|sandwich)(\s+please)?\.?$',
            r'i\'?m\s+(very\s+|so\s+)?hungry',
            r'when\s+(is|are)\s+(the\s+)?(meal|food|lunch|dinner|breakfast)\s+(served|coming)',
            # Spanish
            r'(puedo\s+tener|me\s+trae|quiero|necesito)\s*(un|una|algo\s+de)?\s*(comida|almuerzo|cena|bocadillo|snack)',
            r'tengo\s+(mucha\s+)?hambre',
            r'cuándo\s+(es|sirven)\s+(la\s+)?(comida|almuerzo|cena)',
            # French
            r'(puis-je\s+avoir|je\s+voudrais|apportez-moi)\s*(de\s+la|un|une)?\s*(nourriture|repas|collation|sandwich)',
            r'j\'?ai\s+(très\s+)?faim',
            r'quand\s+(est|sert-on)\s+(le\s+)?(repas|déjeuner|dîner)',
            # German
            r'(kann\s+ich|ich\s+möchte|bringen\s+sie\s+mir)\s*(ein|eine|etwas)?\s*(essen|mahlzeit|snack|sandwich)',
            r'ich\s+habe\s+(großen\s+)?hunger',
            r'wann\s+(ist|gibt\s+es)\s+(das\s+)?(essen|mittagessen|abendessen)',
            # Portuguese
            r'(posso\s+ter|me\s+traz|quero|preciso)\s*(um|uma|algo)?\s*(comida|refeição|lanche|sanduíche)',
            r'estou\s+com\s+(muita\s+)?fome',
            r'quando\s+(é|serve)\s+(a\s+)?(comida|refeição|almoço|jantar)',
            # Hindi
            r'(खाना|भोजन|नाश्ता)\s*(चाहिए|दीजिए|लाइए)',
            r'(मुझे\s+)?भूख\s+लगी',
            r'खाना\s+कब\s+(है|आएगा|मिलेगा)',
            # Thai
            r'(ขอ|ต้องการ|อยากได้)\s*(อาหาร|ข้าว|ของว่าง)',
            r'หิว(ข้าว)?',
            r'อาหาร\s*(เสิร์ฟ)?\s*กี่โมง',
        ]
    },
    'blanket': {
        'patterns': [
            # English
            r'(can\s+i\s+(get|have)|i\'?d?\s+like|bring\s+me|i\s+need|i\s+want|please\s+get)\s*(a\s+)?(blanket|cover)',
            r'i\'?m\s+(very\s+|so\s+)?(cold|freezing|chilly)',
            r'it\'?s\s+(too\s+)?(cold|freezing|chilly)',
            # Spanish
            r'(puedo\s+tener|me\s+trae|quiero|necesito)\s*(una\s+)?(manta|cobija|frazada)',
            r'tengo\s+(mucho\s+)?frío',
            r'hace\s+(mucho\s+)?frío',
            # French
            r'(puis-je\s+avoir|je\s+voudrais|apportez-moi)\s*(une\s+)?(couverture)',
            r'j\'?ai\s+(très\s+)?froid',
            r'il\s+fait\s+(très\s+)?froid',
            # German
            r'(kann\s+ich|ich\s+möchte|bringen\s+sie\s+mir)\s*(eine\s+)?(decke)',
            r'mir\s+ist\s+(sehr\s+)?kalt',
            r'es\s+ist\s+(sehr\s+)?kalt',
            # Portuguese
            r'(posso\s+ter|me\s+traz|quero|preciso)\s*(um\s+|uma\s+)?(cobertor|manta)',
            r'estou\s+com\s+(muito\s+)?frio',
            r'está\s+(muito\s+)?frio',
            # Hindi
            r'(कंबल|चादर)\s*(चाहिए|दीजिए|लाइए)',
            r'(मुझे\s+)?ठंड\s+लग\s+रही',
            r'बहुत\s+ठंड\s+है',
            # Thai
            r'(ขอ|ต้องการ|อยากได้)\s*(ผ้าห่ม)',
            r'หนาว|เย็น',
        ]
    },
    'pillow': {
        'patterns': [
            # English
            r'(can\s+i\s+(get|have)|i\'?d?\s+like|bring\s+me|i\s+need|i\s+want|please\s+get)\s*(a\s+)?(pillow|cushion|headrest)',
            # Spanish
            r'(puedo\s+tener|me\s+trae|quiero|necesito)\s*(una\s+)?(almohada|cojín)',
            # French
            r'(puis-je\s+avoir|je\s+voudrais|apportez-moi)\s*(un\s+)?(oreiller|coussin)',
            # German
            r'(kann\s+ich|ich\s+möchte|bringen\s+sie\s+mir)\s*(ein\s+)?(kissen|kopfkissen)',
            # Portuguese
            r'(posso\s+ter|me\s+traz|quero|preciso)\s*(um\s+|uma\s+)?(travesseiro|almofada)',
            # Hindi
            r'(तकिया)\s*(चाहिए|दीजिए|लाइए)',
            # Thai
            r'(ขอ|ต้องการ|อยากได้)\s*(หมอน)',
        ]
    },
    'assistance': {
        'patterns': [
            # English
            r'(help\s+me|assist\s+me|call\s*(the\s+)?(attendant|crew|steward)|i\s+need\s+(help|assistance)|emergency)',
            # Spanish
            r'(ayúdeme|ayuda|llame\s+(al\s+)?(auxiliar|tripulación)|necesito\s+ayuda|emergencia)',
            # French
            r'(aidez-moi|aide|appelez\s+(l\'?)?(hôtesse|équipage)|j\'?ai\s+besoin\s+d\'?aide|urgence)',
            # German
            r'(helfen\s+sie\s+mir|hilfe|rufen\s+sie\s+(die\s+)?(flugbegleiter|besatzung)|ich\s+brauche\s+hilfe|notfall)',
            # Portuguese
            r'(me\s+ajude|ajuda|chame\s+(o\s+|a\s+)?(comissário|tripulação)|preciso\s+de\s+ajuda|emergência)',
            # Hindi
            r'(मदद|सहायता)\s*(चाहिए|कीजिए)|इमरजेंसी|आपातकाल',
            # Thai
            r'(ช่วย|ต้องการความช่วยเหลือ|เรียก\s*(พนักงาน)?|ฉุกเฉิน)',
        ]
    },
    'medical': {
        'patterns': [
            # English
            r'(i\'?m\s+|i\s+feel\s+|feeling\s+)(sick|ill|nauseous|dizzy|faint|unwell)',
            r'i\s+(have|need)\s+(a\s+)?(headache|pain|medicine|medication|doctor|medical)',
            r'not\s+feeling\s+well',
            # Spanish
            r'(me\s+siento|estoy)\s+(enfermo|mareado|mal)',
            r'(tengo|necesito)\s+(dolor|medicina|medicamento|médico)',
            r'no\s+me\s+siento\s+bien',
            # French
            r'(je\s+me\s+sens|je\s+suis)\s+(malade|nauséeux|étourdi|mal)',
            r'(j\'?ai|j\'?ai\s+besoin\s+d\'?)\s*(mal|douleur|médicament|médecin)',
            r'je\s+ne\s+me\s+sens\s+pas\s+bien',
            # German
            r'(ich\s+fühle\s+mich|mir\s+ist)\s+(schlecht|übel|schwindelig|unwohl)',
            r'(ich\s+habe|ich\s+brauche)\s+(kopfschmerzen|schmerzen|medikament|arzt)',
            r'mir\s+geht\s+es\s+nicht\s+gut',
            # Portuguese
            r'(estou\s+me\s+sentindo|estou)\s+(doente|enjoado|tonto|mal)',
            r'(tenho|preciso\s+de)\s+(dor|remédio|medicamento|médico)',
            r'não\s+(estou\s+)?me\s+sentindo\s+bem',
            # Hindi
            r'(मुझे\s+)?(बीमार|चक्कर|मिचली|तबीयत\s+खराब)',
            r'(दर्द|दवा|डॉक्टर)\s*(है|चाहिए)',
            r'तबीयत\s+ठीक\s+नहीं',
            # Thai
            r'(รู้สึก|เป็น)\s*(ไม่สบาย|คลื่นไส้|เวียนศีรษะ|ป่วย)',
            r'(ปวด|ต้องการ\s*ยา|หมอ)',
            r'ไม่ค่อยสบาย',
        ]
    },
    'entertainment': {
        'patterns': [
            # English
            r'(can\s+i\s+(get|have)|i\'?d?\s+like|bring\s+me|how\s+do\s+i)\s*(a\s+|some\s+)?(headphone|headset|earphone)',
            r'how\s+(do\s+i|to)\s+(use|access|connect)\s+(the\s+)?(wifi|wi-fi|internet|entertainment|movie|screen)',
            r'(wifi|wi-fi|internet)\s+(password|not working|down)',
            # Spanish
            r'(puedo\s+tener|me\s+trae|cómo\s+uso)\s*(unos\s+)?(auriculares|audífonos)',
            r'cómo\s+(uso|accedo|conecto)\s+(el\s+)?(wifi|internet|entretenimiento|película|pantalla)',
            r'(wifi|internet)\s+(contraseña|no\s+funciona)',
            # French
            r'(puis-je\s+avoir|je\s+voudrais|comment\s+utiliser)\s*(des\s+)?(écouteurs|casque)',
            r'comment\s+(utiliser|accéder|connecter)\s+(le\s+|au\s+)?(wifi|internet|divertissement|film|écran)',
            r'(wifi|internet)\s+(mot\s+de\s+passe|ne\s+fonctionne\s+pas)',
            # German
            r'(kann\s+ich|ich\s+möchte|wie\s+benutze\s+ich)\s*(einen\s+)?(kopfhörer|headset)',
            r'wie\s+(benutze|verbinde)\s+ich\s+(das\s+)?(wifi|wlan|internet|unterhaltung|film|bildschirm)',
            r'(wifi|wlan|internet)\s+(passwort|funktioniert\s+nicht)',
            # Portuguese
            r'(posso\s+ter|me\s+traz|como\s+uso)\s*(uns\s+)?(fones|headphone)',
            r'como\s+(uso|acesso|conecto)\s+(o\s+)?(wifi|internet|entretenimento|filme|tela)',
            r'(wifi|internet)\s+(senha|não\s+funciona)',
            # Hindi
            r'(हेडफोन|ईयरफोन)\s*(चाहिए|दीजिए)',
            r'(wifi|वाईफाई|इंटरनेट)\s*(कैसे|पासवर्ड|नहीं\s+चल\s+रहा)',
            # Thai
            r'(ขอ|ต้องการ)\s*(หูฟัง)',
            r'(wifi|ไวไฟ|อินเทอร์เน็ต)\s*(ใช้ยังไง|รหัส|ไม่ทำงาน)',
        ]
    },
}


def detect_service_request(query: str) -> Optional[Dict[str, str]]:
    """Detect if user is requesting in-flight services using multilingual pattern matching.
    
    Supports: English, Spanish, French, German, Portuguese, Hindi, Thai
    
    Distinguishes between:
    - Service requests: "Can I get some water?" "Tengo sed" "J'ai froid"
    - Travel queries: "What can I eat in Japan?" "¿Qué puedo comer en Japón?"
    
    Returns:
        dict with serviceType, priority, message if service request detected
        None if this is a travel query or no service pattern matched
    """
    query_lower = query.lower()
    
    # First check for travel indicators - skip service detection for travel queries
    for pattern in TRAVEL_INDICATORS:
        if re.search(pattern, query_lower):
            return None  # Travel query, not a service request
    
    # Check service patterns
    for service_type, config in SERVICE_PATTERNS.items():
        for pattern in config['patterns']:
            if re.search(pattern, query_lower):
                priority = 'high' if service_type == 'medical' else 'medium'
                if service_type == 'entertainment':
                    priority = 'low'
                
                return {
                    'serviceType': service_type,
                    'priority': priority,
                    'message': query
                }
    
    return None
