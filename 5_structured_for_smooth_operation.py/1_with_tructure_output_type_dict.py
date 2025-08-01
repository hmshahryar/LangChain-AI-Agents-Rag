from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from typing import Annotated, Optional, Literal

load_dotenv()
api_of_gemini = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_of_gemini)

class Review(BaseModel): # annotation to properly explain the things correctly 
    key_themes :Annotated[list[str], "Write all the key themes discuss in reviews in a list "]
    pros:Annotated[Optional[list[str]] , "write down all the pros in side the list "]
    cons:Annotated[Optional[list[str]], "write down all the cons inide the list"]
    summary: Annotated[str,"A berief summary of the review"]
    sentiments: Annotated[Literal["positive","negative"] , "rethurn the sentiment positive or negative "]
    name: Annotated[Optional[str], "write the name and age with coma seprate as a list of the customer "]

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""
As someone who practically lives at airports, I've had my fair share of experiences at airline counters. Here's my take on what it's really like from a customer's point of view.

Key Things I Notice & Expect
When I approach an airline counter, I'm looking for a few core things:

Speed: Can they get me through quickly, especially when I'm running late or have a tight connection?

Helpfulness: Are the staff genuinely willing to assist, or do they just want to move me along?

Waiting Game: How long is that queue going to be? It's often the make-or-break for my pre-flight mood.

Clear Information: I need straightforward answers about my flight, baggage, or any changes. No jargon, please.

Smooth Process: Does it feel like a well-oiled machine, or a chaotic free-for-all?

Tech Integration: While I'm at the counter, I appreciate it if they can quickly access my details, rather than having to re-enter everything.

What I Appreciate (The Pros)
Despite the potential for queues, there are definite upsides to dealing with a human at the counter:

Real Solutions: This is where you go for real problems. When online check-in fails, my flight is cancelled, or I have a last-minute change, the counter staff are often the only ones who can actually fix it. They can rebook me, waive fees, or find alternatives that a machine never could.

Awkward Baggage & Extras: Have an oversized bag? Need to add a pet or special equipment? This is where the human touch is invaluable. They handle the tricky stuff that kiosks can't, ensuring my luggage gets tagged correctly and smoothly.

Peace of Mind with Documents: Especially for international trips, having an agent visually check my passport, visa, and other documents gives me huge peace of mind. It saves potential headaches at security or immigration later on.

Genuine Empathy: When travel goes sideways, a calm, empathetic agent can make all the difference. Their ability to listen and reassure you when you're stressed is a major plus.

What Drives Me Nuts (The Cons)
Unfortunately, the downsides often outweigh the benefits for routine tasks:

The Endless Queue: This is the absolute worst. I've wasted countless hours standing in line, especially during peak travel times or after flight disruptions. It's frustrating when you know your issue could have been resolved in minutes.

Hit-or-Miss Service: Some agents are superstars – efficient, friendly, and resourceful. Others can be slow, uncommunicative, or seem like they'd rather be anywhere else. The inconsistency is annoying.

Limited Availability: At smaller airports or very early/late flights, sometimes the counter isn't even open, forcing you to rely on less personal methods or just wait.

Outdated Systems: Sometimes it feels like they're still using technology from the last century. Manual input and slow system responses can make a simple task take forever.

The Bottleneck Effect: Even with plenty of staff, sometimes the physical layout or the sheer volume of people makes the counter area feel like a chaotic bottleneck, especially when everyone has huge suitcases.

My Overall Takeaway
For a frequent traveler like me, the airline counter is a necessary evil. I prefer to do everything online or at a kiosk if possible, just to avoid the queues. However, when things go wrong, or I have an unusual request, the human behind the counter is invaluable. Airlines really need to focus on improving efficiency and ensuring consistent, high-quality customer service at these points to truly enhance the passenger experience.
""")

print(result.sentiments)

print(result)