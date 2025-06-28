from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy


embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-exp-03-07",dimensions= 380)

documents =  [
    "The Ababeel missile is a medium-range ballistic missile (MRBM) developed by Pakistan, notable for its MIRV (Multiple Independently targetable Reentry Vehicles) capability. With an approximate range of 2,200 kilometers, Ababeel can deliver multiple nuclear warheads to different targets simultaneously, making it a key component in Pakistan’s second-strike deterrence strategy. Tested successfully in 2017, the missile is land-based and enhances Pakistan's ability to penetrate Indian missile defense systems.",

    "Shaheen-I is a short-range ballistic missile (SRBM) with a range of up to 750 kilometers. It is road-mobile and capable of carrying either nuclear or conventional warheads. Operational in Pakistan’s strategic forces, it is designed for quick deployment and regional tactical use. Shaheen-I plays a critical role in maintaining a credible deterrence posture against neighboring threats, particularly from India.",

    "Shaheen-II is a medium-range ballistic missile (MRBM) with an estimated range of around 1,500 kilometers. It is capable of delivering nuclear warheads and is considered more advanced than Shaheen-I in terms of accuracy and range. The missile is road-mobile and has been operationally deployed. Its primary role is strategic deterrence, capable of striking deep into Indian territory if necessary.",

    "Shaheen-III is Pakistan’s longest-range missile to date, with a reach of approximately 2,750 kilometers. It is an intermediate-range ballistic missile (IRBM) developed to counter threats from Indian naval bases in the Andaman and Nicobar Islands. With nuclear capability and high accuracy, it demonstrates Pakistan’s intent to ensure full-spectrum deterrence, especially in maritime zones.",

    "The Ghauri missile is a medium-range ballistic missile (MRBM) with a range of approximately 1,300 kilometers. Developed with assistance from North Korea’s Nodong missile, it uses liquid fuel and can carry both nuclear and conventional warheads. Ghauri was among Pakistan’s earliest MRBMs and was designed to ensure strategic parity with India's missile capabilities. It remains in service but has been supplemented by more advanced systems like the Shaheen series.",

    "Nasr is a tactical ballistic missile with a very short range of 70 kilometers. It is road-mobile and designed to carry low-yield nuclear warheads. The primary goal of Nasr is to serve as a battlefield deterrent, particularly aimed at countering India's Cold Start Doctrine. Its rapid deployment and nuclear capability make it a vital part of Pakistan’s tactical nuclear weapon doctrine.",

    "Babur (Hatf-VII) is a subsonic cruise missile with a range of 450 to 700 kilometers, capable of carrying nuclear or conventional warheads. It is launched from land, sea, and potentially air platforms. Babur uses terrain-hugging flight and low radar visibility to evade defenses. Its precision-strike capability makes it suitable for both strategic and tactical roles in Pakistan's missile doctrine.",

    "Ra'ad is an air-launched cruise missile (ALCM) designed for use with Pakistani fighter aircraft like the JF-17 and Mirage III. With a range of around 350 kilometers, it is capable of delivering nuclear or conventional payloads. Ra’ad provides Pakistan with second-strike capability from the air, giving strategic flexibility and survivability.",

    "Hatf-I is Pakistan's first-generation short-range ballistic missile with a range of 70 to 100 kilometers. Initially developed as an artillery rocket, it can carry conventional warheads and was later upgraded. Though largely outdated now, Hatf-I played a foundational role in Pakistan’s early missile development programs.",

    "Hatf-III Ghaznavi is a short-range ballistic missile with a range of about 290 kilometers, capable of delivering conventional and nuclear warheads. It is road-mobile and designed for rapid deployment. Ghaznavi is part of Pakistan’s effort to maintain regional deterrence through a range of short- to medium-range missile options."
]
query = 'tel me missile capable for above 2000 km '

doc_emb = embedding.embed_documents(documents)
query_embed = embedding.embed_query(query)
index , scors=cosine_similarity([query_embed], doc_emb)[0]
sorted(list(enumerate(scors)),key=lambda x : x[1])[-1]

print(query)
print(documents[index])
print("similirly score is : " , scors)