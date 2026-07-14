from app.agents.router import AgentRouter

router = AgentRouter()

response = router.route(

    question="Why is my servo overheating?",

    product="MG90S"

)

print(response)