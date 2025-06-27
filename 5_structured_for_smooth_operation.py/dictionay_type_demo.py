from typing import TypedDict , Optional
from pydantic import BaseModel,EmailStr , Field
class person(TypedDict):

    name:str
    age:int

new_dict:person = {"name" : "shahryar", "age":19}
# print(new_dict)

class Auth(BaseModel):

    cgpa : float = Field(gt=2 , lt=10 ,description="a valsue represent the cgpa ")
    # email: EmailStr
    age: Optional[int] = None
    name:str = 'shahryar'

new_student = {"name":"sahri" , 'cgpa':8 , 'age':10}    
new_ = Auth(**new_student)
# print(new_)
#emailoistr neede to be insatled in pc when run  
pydantic_to_dict = dict(new_)
# print(pydantic_to_dict)

pydantic_to_json = new_.model_dump_json() 
print(pydantic_to_json)
print(type(pydantic_to_json))