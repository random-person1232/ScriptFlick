from collections import defaultdict
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from typing import Optional, Dict, List, Tuple
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class EmojiMapper:
    def __init__(self):
        
        try:
            
            # Initialize stopwords and lemmatizer
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
        except Exception as e:
            logger.error(f"Failed to initialize EmojiMapper: {e}")
            raise RuntimeError(f"EmojiMapper initialization failed: {e}")
        
        # Custom emoji mappings for common concepts
        self.custom_mappings = {
    # Emotions
    'happy': '😊', 'sad': '😢', 'angry': '😠', 'love': '❤️',
    'laugh': '😄', 'cry': '😢', 'smile': '😊', 'joy': '😊',
    'fear': '😨', 'surprise': '😲', 'confused': '😕', 'bored': '😐',
    'excited': '🤩', 'proud': '😌', 'shy': '☺️', 'tired': '😴',
    'disappointed': '😞', 'frustrated': '😣', 'sick': '🤒', 'nervous': '😬',
    'grateful': '🙏', 'hopeful': '🤞', 'lonely': '😔', 'sorry': '🙏',
    'relaxed': '😌', 'sleepy': '😴', 'blessed': '😇', 'cool': '😎',
    'kiss': '😘', 'wink': '😉', 'silly': '😜', 'crazy': '🤪',
    'shocked': '😱', 'thinking': '🤔', 'mind blown': '🤯', 'sweat': '😓',
    'scream': '😱', 'zzz': '💤', 'celebrate': '🥳', 'content': '😊',
    'determined': '😤', 'annoyed': '😒', 'ashamed': '😳', 'greedy': '🤑',
    'speechless': '😶', 'sneezing': '🤧', 'lying': '🤥', 'nerd': '🤓',

    # Nature
    'sun': '☀️', 'moon': '🌙', 'star': '⭐', 'water': '💧',
    'fire': '🔥', 'mountain': '⛰️', 'tree': '🌳', 'flower': '🌸',
    'forest': '🌲', 'ocean': '🌊', 'rain': '🌧️', 'snow': '❄️',
    'wind': '💨', 'cloud': '☁️', 'rainbow': '🌈', 'island': '🏝️',
    'desert': '🏜️', 'volcano': '🌋', 'earth': '🌍', 'leaf': '🍃',
    'butterfly': '🦋', 'sunrise': '🌅', 'sunset': '🌇', 'river': '🏞️',
    'cactus': '🌵', 'bamboo': '🎍', 'palm': '🌴', 'mushroom': '🍄',
    'maple leaf': '🍁', 'rose': '🌹', 'tulip': '🌷', 'sunflower': '🌻',
    'herb': '🌿', 'seedling': '🌱', 'evergreen tree': '🌲', 'deciduous tree': '🌳',
    'four leaf clover': '🍀', 'earthquake': '🌋', 'tsunami': '🌊', 'meteoroid': '☄️',

    # Animals
    'cat': '🐱', 'dog': '🐶', 'bird': '🐦', 'fish': '🐟',
    'horse': '🐴', 'cow': '🐮', 'pig': '🐷', 'sheep': '🐑',
    'lion': '🦁', 'tiger': '🐯', 'bear': '🐻', 'elephant': '🐘',
    'monkey': '🐒', 'rabbit': '🐰', 'panda': '🐼', 'koala': '🐨',
    'kangaroo': '🦘', 'penguin': '🐧', 'frog': '🐸', 'whale': '🐳',
    'dolphin': '🐬', 'shark': '🦈', 'crab': '🦀', 'octopus': '🐙',
    'snail': '🐌', 'snake': '🐍', 'turtle': '🐢', 'crocodile': '🐊',
    'bat': '🦇', 'sloth': '🦥', 'dinosaur': '🦖', 'unicorn': '🦄',
    'llama': '🦙', 'giraffe': '🦒', 'hippopotamus': '🦛', 'rhinoceros': '🦏',
    'otter': '🦦', 'swan': '🦢', 'peacock': '🦚', 'parrot': '🦜',
    'lobster': '🦞', 'mosquito': '🦟', 'microbe': '🦠',

    # Plants & Trees
    'cactus': '🌵', 'palm tree': '🌴', 'pine tree': '🌲', 'deciduous tree': '🌳',
    'herb': '🌿', 'shamrock': '☘️', 'four leaf clover': '🍀', 'maple leaf': '🍁',
    'fallen leaf': '🍂', 'leaf fluttering in wind': '🍃', 'mushroom': '🍄',
    'sheaf of rice': '🌾', 'bouquet': '💐', 'cherry blossom': '🌸',
    'rose': '🌹', 'hibiscus': '🌺', 'sunflower': '🌻', 'tulip': '🌷',

    # Astronomy & Space
    'sun': '☀️', 'moon': '🌙', 'star': '⭐', 'comet': '☄️',
    'meteor': '☄️', 'milky way': '🌌', 'satellite': '🛰️', 'rocket': '🚀',
    'telescope': '🔭', 'alien': '👽', 'ufo': '🛸', 'astronaut': '👩‍🚀',
    'earth': '🌍', 'planet': '🪐', 'black hole': '⚫', 'galaxy': '🌌',
    'constellation': '✨', 'space': '🌌', 'eclipse': '🌑', 'astronomy': '🔭',

    # Famous Buildings & Landmarks
    'statue of liberty': '🗽', 'eiffel tower': '🗼', 'big ben': '🕰️',
    'colosseum': '🏛️', 'pyramids': '🗿', 'taj mahal': '🕌',
    'great wall': '🧱', 'sydney opera house': '🎭', 'mount fuji': '🗻',
    'leaning tower of pisa': '🏛️', 'christ the redeemer': '⛪',
    'golden gate bridge': '🌉', 'burj khalifa': '🏙️', 'petronas towers': '🏙️',
    'tower bridge': '🌉', 'niagara falls': '💦', 'machu picchu': '⛰️',
    'angkor wat': '🕌', 'london eye': '🎡', 'times square': '🌆',

    # Music Genres
    'rock': '🎸', 'pop': '🎤', 'classical': '🎻', 'jazz': '🎷',
    'hip hop': '🎧', 'country': '🤠', 'electronic': '🎹', 'reggae': '🎶',
    'blues': '🎺', 'metal': '🤘', 'folk': '🪕', 'opera': '🎭',

    # Fitness & Wellness
    'gym': '🏋️', 'yoga': '🧘', 'meditation': '🧘', 'running': '🏃',
    'cycling': '🚴', 'swimming': '🏊', 'pilates': '🤸', 'aerobics': '🕺',
    'diet': '🥗', 'wellness': '💆', 'spa': '💆', 'massage': '💆',
    'nutrition': '🥑', 'health': '❤️', 'sleep': '😴', 'relaxation': '🧘',

    # Hobbies
    'painting': '🎨', 'photography': '📷', 'gardening': '🌿', 'cooking': '🍳',
    'baking': '🥧', 'knitting': '🧶', 'writing': '✍️', 'reading': '📚',
    'traveling': '✈️', 'gaming': '🎮', 'hiking': '🥾', 'fishing': '🎣',
    'bird watching': '🐦', 'collecting': '🧳', 'dancing': '💃', 'singing': '🎤',
    'crafting': '✂️', 'origami': '📄', 'puzzles': '🧩', 'woodworking': '🪓',

    # Games & Pastimes
    'chess': '♟️', 'cards': '🃏', 'poker': '🎴', 'billiards': '🎱',
    'darts': '🎯', 'bowling': '🎳', 'board games': '🎲', 'video games': '🎮',
    'gambling': '🎰', 'dominoes': '🁣', 'jigsaw puzzle': '🧩', 'kites': '🪁',

    # Festivals & Events
    'new year': '🎉', 'christmas': '🎄', 'easter': '🐰', 'halloween': '🎃',
    'thanksgiving': '🦃', 'valentine': '💘', 'wedding': '💒', 'graduation': '🎓',
    'birthday': '🎂', 'carnival': '🎭', 'parade': '🎉', 'fireworks': '🎆',
    'festival': '🎪', 'concert': '🎵', 'theater': '🎭', 'party': '🥳',

    # Environmental Issues
    'recycle': '♻️', 'pollution': '🏭', 'global warming': '🌡️', 'deforestation': '🌳❌',
    'renewable energy': '💡', 'solar power': '☀️', 'wind power': '💨',
    'water conservation': '💧', 'earthquake': '🌋', 'flood': '🌊', 'drought': '🌵',
    'endangered species': '🐼', 'ocean pollution': '🌊🏭',

    # Spirituality & Religion
    'church': '⛪', 'mosque': '🕌', 'temple': '🏯', 'synagogue': '🕍',
    'prayer': '🙏', 'meditation': '🧘', 'cross': '✝️', 'crescent': '☪️',
    'om': '🕉️', 'star of david': '✡️', 'wheel of dharma': '☸️',
    'yin yang': '☯️', 'peace': '☮️', 'menorah': '🕎', 'hamsa': '🪬',

    # Art & Culture
    'museum': '🏛️', 'art': '🎨', 'theater': '🎭', 'dance': '💃',
    'music': '🎵', 'literature': '📖', 'sculpture': '🗿', 'film': '🎬',
    'photography': '📷', 'poetry': '📜', 'opera': '🎼', 'ballet': '🩰',
    'comedy': '😂', 'drama': '🎭', 'festival': '🎪', 'exhibition': '🖼️',

    # Additional Occupations
    'judge': '👨‍⚖️', 'pilot': '👨‍✈️', 'scientist': '👩‍🔬', 'engineer': '👷',
    'chef': '👨‍🍳', 'artist': '👩‍🎨', 'photographer': '📷', 'writer': '✍️',
    'dentist': '🦷', 'veterinarian': '🐾', 'architect': '📐', 'journalist': '📝',
    'flight attendant': '👩‍✈️', 'salesperson': '🛍️', 'pharmacist': '💊',
    'psychologist': '🧠', 'astronaut': '👩‍🚀', 'paramedic': '🚑', 'plumber': '🛠️',

    # Additional Technology
    'smartphone': '📱', 'tablet': '💻', 'drone': '🛸', 'smartwatch': '⌚',
    '3d printer': '🖨️', 'vr headset': '🎮', 'robot': '🤖', 'usb': '💾',
    'cloud computing': '☁️', 'internet': '🌐', 'server': '🖥️', 'code': '💻',
    'wifi': '📶', 'bluetooth': '📲', 'satellite': '🛰️', 'email': '📧',
    'artificial intelligence': '🤖', 'bitcoin': '💰', 'blockchain': '⛓️',

    # Additional Transportation
    'spaceship': '🚀', 'skateboard': '🛹', 'roller skate': '🛼',
    'hoverboard': '🛹', 'canoe': '🛶', 'submarine': '🚢', 'ambulance': '🚑',
    'fire engine': '🚒', 'police car': '🚓', 'taxi': '🚕', 'tractor': '🚜',
    'kick scooter': '🛴', 'bus': '🚌', 'tram': '🚊', 'metro': '🚇',

    # Additional Food & Drink
    'pancake': '🥞', 'bacon': '🥓', 'sandwich': '🥪', 'taco': '🌮',
    'popcorn': '🍿', 'chocolate': '🍫', 'cookie': '🍪', 'doughnut': '🍩',
    'pretzel': '🥨', 'bagel': '🥯', 'croissant': '🥐', 'butter': '🧈',
    'shrimp': '🍤', 'steak': '🥩', 'spaghetti': '🍝', 'soup': '🥣',
    'salad': '🥗', 'sushi': '🍣', 'ramen': '🍜', 'curry': '🍛',
    'lobster': '🦞', 'crab': '🦀', 'cheeseburger': '🍔', 'hot dog': '🌭',
    'pizza': '🍕', 'cake': '🍰', 'ice cream': '🍨', 'pie': '🥧',

    # Additional Sports & Activities
    'baseball': '⚾', 'basketball': '🏀', 'soccer': '⚽', 'football': '🏈',
    'tennis': '🎾', 'volleyball': '🏐', 'rugby': '🏉', 'golf': '⛳',
    'cricket': '🏏', 'table tennis': '🏓', 'badminton': '🏸', 'boxing': '🥊',
    'martial arts': '🥋', 'skiing': '⛷️', 'snowboarding': '🏂', 'surfing': '🏄',
    'horse riding': '🏇', 'fencing': '🤺', 'archery': '🏹', 'ice skating': '⛸️',
    'weightlifting': '🏋️', 'diving': '🤿', 'kayaking': '🚣', 'rock climbing': '🧗',

    # Additional Concepts
    'innovation': '💡', 'growth': '🌱', 'balance': '⚖️', 'connection': '🌐',
    'community': '👥', 'diversity': '🌈', 'equality': '🟰', 'opportunity': '🚪',
    'knowledge': '📚', 'power': '💪', 'wealth': '💰', 'happiness': '😊',
    'motivation': '🚀', 'creativity': '🎨', 'leadership': '👑', 'teamwork': '🤝',
    'education': '🎓', 'environment': '🌳', 'security': '🔒', 'technology': '💻',

    # Additional Countries & Flags
    'argentina': '🇦🇷', 'belgium': '🇧🇪', 'colombia': '🇨🇴', 'denmark': '🇩🇰',
    'egypt': '🇪🇬', 'finland': '🇫🇮', 'greece': '🇬🇷', 'hungary': '🇭🇺',
    'ireland': '🇮🇪', 'jamaica': '🇯🇲', 'kenya': '🇰🇪', 'luxembourg': '🇱🇺',
    'morocco': '🇲🇦', 'netherlands': '🇳🇱', 'norway': '🇳🇴', 'philippines': '🇵🇭',
    'qatar': '🇶🇦', 'sweden': '🇸🇪', 'turkey': '🇹🇷', 'ukraine': '🇺🇦',
    'vietnam': '🇻🇳', 'yemen': '🇾🇪', 'zimbabwe': '🇿🇼',

    # Additional Weather
    'tornado': '🌪️', 'cyclone': '🌀', 'fog': '🌫️', 'hail': '🌨️',
    'meteor shower': '☄️', 'full moon': '🌕', 'new moon': '🌑', 'crescent moon': '🌙',

    # Additional Feelings & Emotions
    'anxious': '😰', 'bored': '😒', 'sleepy': '😴', 'surprised': '😲',
    'jealous': '😒', 'embarrassed': '😳', 'suspicious': '🤨', 'relieved': '😌',
    'skeptical': '🤔', 'joyful': '😁', 'ecstatic': '😆', 'optimistic': '😃',

    # Additional Modes of Transportation
    'unicycle': '🚲', 'monorail': '🚝', 'airship': '🛩️', 'gondola': '🚠',
    'cable car': '🚡', 'zeppelin': '🛩️', 'rickshaw': '🛺', 'trolleybus': '🚎',

    # Additional Mathematical Symbols
    'plus': '+', 'minus': '-', 'divide': '÷', 'multiply': '×',
    'equal': '=', 'not equal': '≠', 'greater than': '>', 'less than': '<',
    'pi': 'π', 'infinity': '∞', 'percent': '%', 'degree': '°',

    # Additional Scientific Concepts
    'atom': '⚛️', 'dna': '🧬', 'microscope': '🔬', 'telescope': '🔭',
    'satellite': '🛰️', 'battery': '🔋', 'magnet': '🧲', 'chemical': '⚗️',
    'lab': '🔬', 'equation': '➗', 'experiment': '🧪', 'gear': '⚙️',

    # Additional Spiritual & Religious Symbols
    'prayer beads': '📿', 'lotus': '💮', 'ankh': '☥', 'kaaba': '🕋',
    'dhikr': '🕉️', 'mosque': '🕌', 'church': '⛪', 'synagogue': '🕍',

    # Additional Holidays & Celebrations
    'hanukkah': '🕎', 'diwali': '🪔', 'ramadan': '🌙', 'eid': '🕌',
    'chinese new year': '🧧', 'oktoberfest': '🍻', 'cinco de mayo': '🇲🇽',
    'st patricks day': '🍀', 'halloween': '🎃', 'thanksgiving': '🦃',

    # Additional Health & Wellness
    'first aid': '🩹', 'blood': '🩸', 'pill': '💊',
    'stethoscope': '🩺', 'crutch': '🩼', 'wheelchair': '♿', 'hospital': '🏥',
    'mask': '😷', 'x-ray': '🩻',

    # Additional Communication
    'fax': '📠', 'pager': '📟', 'megaphone': '📣', 'microphone': '🎤',
    'loudspeaker': '🔊', 'bell': '🔔', 'postbox': '📮', 'email': '📧',
    'phone': '☎️', 'cell phone': '📱', 'antenna': '📡',
    'brain': '🧠', 'heart': '❤️', 'bones': '🦴', 'lungs': '🫁',
'heart eyes': '😍', 'sleeping face': '😪', 'scream': '😱',
'medal': '🏅', 'trophy': '🏆', 'sparkles': '✨', 'lightning': '⚡',
'tornado': '🌪️', 'rainbow': '🌈', 'ring': '💍', 'crown': '👑',
'skeleton': '💀', 'ghost': '👻', 'angel': '👼', 'zombie': '🧟',
'robot': '🤖', 'vampire': '🧛', 'genie': '🧞', 'fairy': '🧚',
'mermaid': '🧜', 'witch': '🧙', 'wizard': '🧙‍♂️', 'elf': '🧝',
'clown': '🤡', 'exploding head': '🤯', 'relieved': '😌',
'basket': '🧺', 'shopping cart': '🛒', 'coin': '🪙', 'banknote': '💵',
'dollar': '💲', 'handshake': '🤝', 'peace sign': '✌️',
'praying': '🙏', 'writing': '✍️', 'clapping': '👏', 'fist bump': '👊',
'waving': '👋', 'hug': '🤗', 'raising hands': '🙌',
'thinking': '🤔', 'blush': '😊', 'sleeping': '💤', 'fireworks': '🎆',
'spiral': '🌀', 'puzzle piece': '🧩', 'lock': '🔒', 'unlock': '🔓',
'mountain': '🏔️', 'river': '🏞️', 'beach': '🏖️', 'sunset': '🌇',
'violin': '🎻', 'trumpet': '🎺', 'drum': '🥁', 'guitar': '🎸',
'microphone': '🎤', 'movie camera': '🎥', 'projector': '📽️',
'keyboard': '⌨️', 'joystick': '🕹️', 'compass': '🧭', 'map': '🗺️',
'flashlight': '🔦', 'magnifying glass': '🔍', 'lock': '🔒',
'brainstorm': '💡', 'gear': '⚙️', 'diamond': '💎', 'snowboard': '🏂',
'popcorn': '🍿', 'pizza slice': '🍕', 'ice cream': '🍦',
'burrito': '🌯', 'tea': '🍵', 'beer': '🍺', 'champagne': '🍾',
'strawberry': '🍓', 'carrot': '🥕', 'corn': '🌽', 'mango': '🥭',
'surfing': '🏄', 'balloon': '🎈', 'airplane': '✈️', 'sailboat': '⛵',
'subway': '🚇', 'taxi': '🚕', 'speedboat': '🚤', 'helicopter': '🚁',
'stop sign': '🛑', 'bus stop': '🚏', 'train': '🚆',
'cloud with lightning': '🌩️', 'fire': '🔥', 'tsunami': '🌊',
'earthquake': '🌋', 'moon with face': '🌜', 'thermometer': '🌡️',
'hermit crab': '🦀', 'skull': '☠️', 'carousel': '🎠',
'roller coaster': '🎢', 'ferris wheel': '🎡', 'goggles': '🥽',
'whistle': '🔔', 'clipboard': '📋', 'syringe': '💉', 'petri dish': '🧫',
'test tube': '🧪', 'pet': '🐾', 'dove': '🕊️', 'recycle': '♻️',
'computer mouse': '🖱️', 'video camera': '📹', 'book': '📖',
'pencil': '✏️', 'paintbrush': '🖌️', 'light bulb': '💡',
'top hat': '🎩', 'baseball': '⚾', 'soccer ball': '⚽', 'football': '🏈',
'tennis racket': '🎾', 'skateboard': '🛹', 'bowling': '🎳',
'scalpel': '🔪', 'eraser': '🩹', 'sunglasses': '🕶️', 'first aid': '⛑️',
'soap': '🧼', 'toolbox': '🧰', 'seedling': '🌱', 'baby bottle': '🍼',
'broom': '🧹', 'mirror': '🪞', 'credit card': '💳', 'cash register': '🧾',
'fire extinguisher': '🧯', 'rocket': '🚀', 'wrench': '🔧', 'hammer': '🔨',
'drill': '🛠️', 'vacuum': '🧹', 'binoculars': '🔭',
'handshake': '🤝', 'peace symbol': '☮️', 'wheelchair': '♿', 'lightning bolt': '⚡',
'shield': '🛡️', 'radioactive': '☢️', 'biohazard': '☣️', 'fire truck': '🚒',
'police officer': '👮', 'detective': '🕵️', 'guard': '💂', 'construction worker': '👷',
'scientist': '🧑‍🔬', 'artist': '👩‍🎨', 'pilot': '👨‍✈️', 'astronaut': '👩‍🚀',
'teacher': '🧑‍🏫', 'judge': '👩‍⚖️', 'chef': '👩‍🍳', 'plumber': '🪠',
'hiker': '🥾', 'mountaineer': '🧗', 'skier': '⛷️', 'climber': '🧗‍♂️',
'teacher': '📚', 'farmer': '👩‍🌾', 'doctor': '🩺', 'nurse': '💉',
'dj': '🎧', 'drummer': '🥁', 'painter': '🎨', 'sculptor': '🗿',
'firefighter': '🚒', 'paramedic': '🚑', 'mechanic': '🔧', 'beekeeper': '🐝',
'programmer': '💻', 'yoga': '🧘', 'golfer': '🏌️', 'fisherman': '🎣',
'gardener': '🌿', 'baker': '🥖', 'photographer': '📸', 'traveler': '🌍',
'magician': '🧙‍♂️', 'ninja': '🥷', 'pirate': '🏴‍☠️', 'mermaid': '🧜‍♀️',
'dancer': '💃', 'superhero': '🦸‍♀️', 'supervillain': '🦹‍♂️', 'genie': '🧞‍♂️',
'sumo wrestler': '🤼‍♂️', 'medal': '🥇', 'diploma': '📜', 'books': '📚',
'school': '🏫', 'university': '🎓', 'backpack': '🎒', 'microscope': '🔬',
'telescope': '🔭', 'abacus': '🧮', 'gear': '⚙️', 'factory': '🏭',
'bank': '🏦', 'hospital': '🏥', 'library': '📖', 'court': '⚖️',
'stadium': '🏟️', 'amphitheater': '🎭', 'bridge': '🌉', 'castle': '🏰',
'skyline': '🏙️', 'mountain range': '🏔️', 'campground': '🏕️', 'cave': '🕳️',
'cliff': '⛰️', 'waterfall': '🏞️', 'riverbank': '🏞️', 'forest': '🌲',
'beach': '🏖️', 'hot springs': '♨️', 'campsite': '⛺', 'lighthouse': '🗼',
'globe': '🌐', 'satellite': '🛰️', 'moon phases': '🌙', 'solar eclipse': '🌞',
'asteroid': '☄️', 'spaceship': '🚀', 'milky way': '🌌', 'shooting star': '🌠',
'bat': '🦇', 'owl': '🦉', 'wolf': '🐺', 'fox': '🦊',
'hedgehog': '🦔', 'rabbit': '🐰', 'koala': '🐨', 'penguin': '🐧',
'toucan': '🦜', 'flamingo': '🦩', 'peacock': '🦚', 'swan': '🦢',
'parrot': '🦜', 'pigeon': '🕊️', 'butterfly': '🦋', 'ladybug': '🐞',
'bee': '🐝', 'ant': '🐜', 'spider': '🕷️', 'scorpion': '🦂',
'shrimp': '🦐', 'crab': '🦀', 'jellyfish': '🪼', 'lobster': '🦞',
'hammer': '🔨', 'pickaxe': '⛏️', 'chisel': '🔪', 'saw': '🪚',
'toolbox': '🧰', 'paint roller': '🖌️', 'paintbrush': '🎨', 'flashlight': '🔦',
'ring': '💍', 'diamond': '💎', 'gemstone': '💠', 'compass': '🧭',
'map': '🗺️', 'key': '🔑', 'magnifying glass': '🔍', 'hourglass': '⌛',
'clock': '🕰️', 'alarm clock': '⏰', 'stopwatch': '⏱️', 'calendar': '📆',
'bookmark': '🔖', 'book': '📖', 'newspaper': '📰', 'pen': '🖊️',
'pencil': '✏️', 'paint palette': '🎨', 'feather': '🪶', 'ink pen': '🖋️',
'phone': '📞', 'cell phone': '📱', 'pager': '📟', 'fax machine': '📠',
'television': '📺', 'radio': '📻', 'computer': '💻', 'printer': '🖨️',
'camera': '📷', 'camcorder': '📹', 'movie projector': '📽️', 'satellite dish': '📡',
'microscope': '🔬', 'test tube': '🧪', 'petri dish': '🧫', 'syringe': '💉',
'stethoscope': '🩺', 'blood drop': '🩸', 'ambulance': '🚑', 'medic': '🧑‍⚕️',
'hospital': '🏥', 'wheelchair': '♿', 'crutches': '🩼', 'bandaid': '🩹',
'money bag': '💰', 'dollar bills': '💵', 'coin': '🪙', 'credit card': '💳',
'bank building': '🏦', 'receipt': '🧾', 'chart': '📊', 'stock market': '📈',
'balance scale': '⚖️', 'judge hammer': '🔨', 'lock': '🔒', 'unlock': '🔓',
'shield': '🛡️', 'key': '🔑', 'link': '🔗', 'calendar': '📅',
'phone': '☎️', 'cellular': '📶', 'wifi': '📡', 'bluetooth': '📲',
'hammer and wrench': '🛠️', 'toolbox': '🧰', 'cogwheel': '⚙️', 'battery': '🔋',
'plug': '🔌', 'circuit board': '🖥️', 'lightbulb': '💡', 'rocket ship': '🚀',
'brain': '🧠', 'robot face': '🤖', 'alien': '👽', 'hourglass': '⏳',
'traffic light': '🚦', 'stop sign': '🛑', 'exit sign': '🚪', 'no entry': '🚫',
'spiral': '🌀', 'black cat': '🐈‍⬛', 'crystal ball': '🔮', 'magic wand': '🪄',
'skull': '☠️', 'tombstone': '🪦', 'pumpkin': '🎃', 'lantern': '🏮',
'flower': '🌸', 'sunflower': '🌻', 'leaf': '🍃', 'tree': '🌳',
'desert': '🏜️', 'island': '🏝️', 'volcano': '🌋', 'ocean': '🌊',
'cloudy': '☁️', 'snowflake': '❄️', 'rainy': '🌧️', 'stormy': '⛈️',
'tornado': '🌪️', 'fire': '🔥', 'tsunami wave': '🌊', 'sun': '☀️',
'moon': '🌙', 'stars': '⭐', 'mountain peak': '⛰️', 'rainbow': '🌈',
'shooting star': '🌠', 'earth': '🌍', 'satellite orbit': '🛰️', 'rocket launch': '🚀',
'flag': '🏳️', 'red flag': '🚩', 'fire extinguisher': '🧯', 'trash bin': '🗑️',
'recycle symbol': '♻️', 'toilet': '🚽', 'shower': '🚿', 'bathtub': '🛁',
'spiral shell': '🐚', 'anchor': '⚓', 'fountain': '⛲', 'pool': '🏊‍♀️',
'candy': '🍬', 'cupcake': '🧁', 'cookie': '🍪', 'pie': '🥧',
'cereal': '🥣', 'steak': '🥩', 'sushi': '🍣', 'noodles': '🍜',
'salad': '🥗', 'eggplant': '🍆', 'tomato': '🍅', 'potato': '🥔',
'cherry': '🍒', 'avocado': '🥑', 'chicken': '🍗', 'ham': '🍖',
'pizza': '🍕', 'hamburger': '🍔', 'hot dog': '🌭', 'fries': '🍟',
'soda': '🥤', 'wine': '🍷', 'beer': '🍺', 'cocktail': '🍸',
'coffee': '☕', 'juice': '🧃', 'milk': '🥛', 'bread': '🍞',
'bacon': '🥓', 'butter': '🧈', 'pancakes': '🥞', 'cheese': '🧀',
'spaghetti': '🍝', 'soup': '🥣', 'sandwich': '🥪', 'burrito': '🌯',
'popcorn': '🍿', 'ice cream cone': '🍦', 'shaved ice': '🍧', 'milkshake': '🥤'

}

    


    def _get_wordnet_pos(self, word: str, tag: str) -> str:
        """Map POS tag to first character used by WordNet."""
        tag_dict = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }
        return tag_dict.get(tag[0], wordnet.NOUN)

    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate semantic similarity between two words."""
        try:
            # Get synsets for both words
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            
            if not synsets1 or not synsets2:
                return 0.0
            
            # Calculate maximum similarity between any pair of synsets
            max_sim = 0
            for syn1 in synsets1:
                for syn2 in synsets2:
                    try:
                        sim = syn1.path_similarity(syn2)
                        if sim and sim > max_sim:
                            max_sim = sim
                    except Exception as e:
                        logger.debug(f"Error calculating similarity: {e}")
                        continue
                        
            return max_sim
        except Exception as e:
            logger.warning(f"Error in similarity calculation: {e}")
            return 0.0

    def extract_most_relevant_keyword(self, text: str, threshold: float = 0.2) -> Optional[str]:
        """Extract most relevant keyword with improved matching."""
        try:
            # Clean and tokenize text
            text = re.sub(r'[^\w\s]', '', text.lower())
            tokens = word_tokenize(text)
            
            # POS tagging
            tagged = nltk.pos_tag(tokens)
            
            # Filter out stopwords and lemmatize
            words_with_pos = []
            for word, tag in tagged:
                if word not in self.stop_words and len(word) > 2:
                    pos = self._get_wordnet_pos(word, tag)
                    lemma = self.lemmatizer.lemmatize(word, pos=pos)
                    words_with_pos.append((lemma, pos))
            
            # Calculate relevancy scores for each word
            word_scores = defaultdict(float)
            for word, pos in words_with_pos:
                max_similarity = 0
                best_emoji_word = None
                
                for emoji_word in self.custom_mappings.keys():
                    similarity = self._calculate_word_similarity(word, emoji_word)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_emoji_word = emoji_word
                
                if max_similarity >= threshold and best_emoji_word:
                    word_scores[best_emoji_word] = max_similarity
            
            # Sort by similarity score
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_words:
                logger.debug(f"Found keyword {sorted_words[0][0]} with score {sorted_words[0][1]}")
                return sorted_words[0][0]
            return None
            
        except Exception as e:
            logger.error(f"Error extracting keyword: {e}")
            return None

    def map_keyword_to_emoji(self, keyword: Optional[str]) -> Optional[str]:
        """Map keyword to emoji."""
        if not keyword:
            return None
        
        emoji = self.custom_mappings.get(keyword)
        if emoji:
            logger.debug(f"Mapped {keyword} to emoji {emoji}")
        return emoji

def create_emoji_style(video_width: int, video_height: int, font_name: str) -> str:
    """Create ASS style for emoji."""
    emoji_y = int(video_height * 0.45)
    font_size = 72
    return (f"Style: Emoji,{font_name},{font_size},&H00FFFFFF,&H000000FF,"
           f"&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,8,10,10,{emoji_y},1")

def create_emoji_events(phrases: List[Dict], video_width: int, video_height: int) -> str:
    """Create ASS events for emoji overlays."""
    try:
        emoji_mapper = EmojiMapper()
        events = []
        emoji_y = int(video_height * 0.45)
        
        for i, phrase in enumerate(phrases):
            # Combine all words in the phrase
            all_words = " ".join([
                word.word for word in (phrase.get('top_line', []) + phrase.get('bottom_line', []))
            ])
            
            keyword = emoji_mapper.extract_most_relevant_keyword(all_words, threshold=0.2)
            emoji_char = emoji_mapper.map_keyword_to_emoji(keyword)
            
            if emoji_char:
                start_time = phrase['start']
                end_time = phrase['end']
                
                # Add emoji with fade effect
                events.append(
                    f"Dialogue: 2,{format_ass_time(start_time)},{format_ass_time(end_time)},"
                    f"Emoji,,0,0,0,,{{\\fad(200,200)}}"
                    f"{{\\an8}}{{\\pos({video_width//2},{emoji_y})}}{emoji_char}"
                )
                logger.debug(f"Added emoji {emoji_char} for phrase {i+1}")
        
        logger.info(f"Created {len(events)} emoji events")
        return "\n".join(events)
        
    except Exception as e:
        logger.error(f"Error creating emoji events: {e}")
        return ""

def format_ass_time(seconds: float) -> str:
    """Format time for ASS subtitles."""
    try:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        centiseconds = int((secs * 100) % 100)
        secs = int(secs)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"  # Changed seconds to secs
    except Exception as e:
        logger.error(f"Error formatting time: {e}")
        return "0:00:00.00"