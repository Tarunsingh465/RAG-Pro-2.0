from app.services.supervisor_service import SupervisorService


supervisor = SupervisorService()


queries = [
    "how are you?"
]


for query in queries:

    result = supervisor.route(query)

    print("\nQuery:", query)
    print("Route:", result.route)
    print("Confidence:", result.confidence)
    print("Reason:", result.reason)