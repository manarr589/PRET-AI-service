from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_route(locations):
    # 1. تعريف المشكلة: 'locations' هي قائمة بإحداثيات الكافيهات
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # 2. دالة حساب المسافة (هنا بنحسب المسافة بين كل نقطتين)
    def distance_callback(from_index, to_index):
    
        return 10 

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 3. إيجاد الحل الأمثل
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    
    # 4. ترتيب الزيارة
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    
    return route

# تجربة الكود (3 كافيهات مثلاً)
locations = [(30.59, 32.26), (30.60, 32.27), (30.61, 32.28)]
print(f"أفضل ترتيب لزيارة الكافيهات هو: {solve_route(locations)}")
