import random
import math
from scipy.spatial.distance import cityblock
import csv
import pandas as pd

# import regression

class Vehicle:
    def __init__(self, company, id, pos_x, pos_y, status, dist_to_dem, total_occup_dist, total_idle_dist, dest_x,
                 dest_y):
        self.company = company  # company for which vehicle is serving
        self.id = id  # id of vehicle
        self.pos_x = pos_x  # x position of vehicle
        self.pos_y = pos_y  # y position of vehicle
        self.status = status  # status of vehicle - "free", "goes_to_cust", and "occupied"
        self.dist_to_dem = dist_to_dem  # attribute where distance to demand is temporarily stored
        self.total_occup_dist = total_occup_dist  # total run distance of vehicle while being occupied
        self.total_idle_dist = total_idle_dist  # total idle distance of vehicle to pick up a customer
        self.dest_x = dest_x  # x position of destination (position of customer or customer's destination)
        self.dest_y = dest_y  # y position of destination (position of customer or customer's destination)


class Demand:
    def __init__(self, company, id, o_x, o_y, d_x, d_y, req_time, wait_match_time, status, veh_id, idle_dist_dem, occup_dist_dem):
        self.company = company  # company that customer has chosen
        self.id = id  # id of demand request
        self.o_x = o_x  # x position of origin
        self.o_y = o_y  # y position of origin
        self.d_x = d_x  # x position of destination
        self.d_y = d_y  # y position of destination
        self.req_time = req_time  # time of receiving the request
        self.wait_match_time = wait_match_time
        self.status = status  # status of demand - "on hold" (waiting to be matched with a car), "waiting" (for a car to arrive), "in car", "finished", and "rejected" if there are no available cars
        self.veh_id = veh_id  # id of matched vehicle
        self.idle_dist_dem = idle_dist_dem  # distance run by veh to pick up this demand
        self.occup_dist_dem = occup_dist_dem    # distance run by veh to bring this dem to its destination


def initialization_veh(num_veh):
    global id_company
    vehicles = []
    for key in num_veh:  # initialization of vehicles
        for i in range(1, num_veh[
                              key] + 1):  # id of each vehicle is 1000 for uber, 2000 for lyft plus the number of this car in company's fleet
            if key == 'uber':
                id_company = 1000
            vehicles.append(Vehicle(key,
                                    id_company + i,  # i.g. id = 2001 for the car number 1 in the fleet of lyft company
                                    random.randint(0, max_x),  # random assignment of start position
                                    random.randint(0, max_y),
                                    'free',
                                    0,
                                    0,
                                    0,
                                    0,
                                    0))
    return vehicles


def initialization_dem(num_veh, demand_count, discarded_time):
    demands = []
    for key in num_veh:  # initialization of demands
        # for i in range(1, random.randint(2, num_veh[key] + 1)):     #the number of demand for each company is <= the number of cars of this company
        # for i in range(1, num_veh[key] + 1):                       #the number of demand is the same as the number of vehicles
        # for i in range(1, 51):                       #the number of demand is fixed
        # for i in range(1, random.randint(round((num_veh[key] + 2) / 2), num_veh[key] + 1)):  # fleet/2 <= demand <= fleet
        # for i in range(1, round(3 * num_veh[key]) + 1):  # demand = n * fleet
        for i in range(1, round(10 * num_veh[key]) + 1):  # 1 * fleet <= demand <= 9 * fleet
            o_x = random.randint(0, max_x)  # random assignment of origin and destination
            o_y = random.randint(0, max_y)
            d_x = random.randint(0, max_x)
            d_y = random.randint(0, max_y)
            #t = random.randint(0, round(max_time / 2)) # time of request shouldn't be too late for a car to be able to pick up the customer and bring to the destination
            #t = random.randint(0, max_time - discarded_time)
            t = random.randint(0, max_time - 1)
            while (o_x == d_x) and (o_y == d_y):  # avoid that origin and destination is the same point
                o_x = random.randint(0, max_x)
                o_y = random.randint(0, max_y)
            occup_dist_dem = abs(o_x - d_x) + abs(o_y - d_y)
            demands.append(Demand(key,
                                  i,
                                  o_x,
                                  o_y,
                                  d_x,
                                  d_y,
                                  t,
                                  0,
                                  'on hold',
                                  0,
                                  0,
                                  occup_dist_dem))
            if t >= discarded_time:
                demand_count[key] += 1
    return demands, demand_count


def matching(time, num_queued_demands_total, num_queued_demands_instant, num_rejected_demands, discarded_time, idle_dist_to_queued_dem, num_queued_dem_served):
    dump = 0        # dump variable to check if there are any requests in the queue
    for dem in demands:
        if dem.status == 'queue':
            dump += 1       # if there is a queue of demands - the dump var is non-zero
    if time >= discarded_time:
        num_queued_demands_instant.append(dump)

    if dump == 0:       # if no requests in the queue
        for dem in demands:
            if (dem.req_time == time and dem.status == 'on hold') or (dem.req_time < time and dem.status == 'queue'):  # choose a demand where the time of receiving the request is equal to the current time and the status of the request is "on hold" OR the demand that is waiting in the queue
                for car in vehicles:
                    if (car.status == 'free') and (
                            dem.company == car.company):  # look for available cars from the requested company
                        vec1 = [car.pos_x, car.pos_y]
                        vec2 = [dem.o_x, dem.o_y]
                        car.dist_to_dem = cityblock(vec1, vec2)  # calculate for each car the distance to the chosen demand
                veh_sorted = sorted(vehicles,
                                    key=lambda x: x.dist_to_dem)  # sort the vehicles by the distance to the demand
                selected_car = next((x for x in veh_sorted if x.company == dem.company and x.status == 'free'),
                                    None)  # select the nearest car (the first in sorted list) subject to being free and from the requested company, return None if no car selected

                if selected_car is None:
                    if dem.wait_match_time < queue_time:   # if no available car - put demand request in a queue for 'queue_time' min
                        if (dem.wait_match_time == 0) and (time >= discarded_time):    # if it's new request - update the number of requests that have been queued
                            num_queued_demands_total[dem.company] += 1

                        dem.status = 'queue'
                        dem.wait_match_time += 1

                    elif dem.wait_match_time >= queue_time:      # if request is in a queue for more than 'queue_time' min - reject it
                        dem.status = 'rejected'
                        if (time >= discarded_time) and (dem.req_time >= discarded_time):
                            num_rejected_demands[dem.company] += 1

                else:
                    dem.veh_id = selected_car.id  # assign the id of chosen car to the demand
                    dem.status = 'waiting'  # change the status of demand
                    for car in vehicles:
                        car.dist_to_dem = 0  # reset the 'distance to demand' variable
                        if car.id == selected_car.id:
                            car.status = 'goes_to_cust'  # change the status of the car
                            car.dest_x = dem.o_x  # change the car destination coordinates
                            car.dest_y = dem.o_y
                            num_free_veh[car.company] -= 1


    else:                  # if there are requests in the queue
        dem_q = []      # new array for queued demand
        for dem in demands:
            if dem.status == "queue":
                dem_q.append(dem)
        dem_sorted = sorted(dem_q, key=lambda x: x.req_time)    # sort queued demand according to the request time from smallest to highest
        for d in dem_sorted:                        # for sorted queued demand
            for car in vehicles:
                if (car.status == 'free') and (
                        d.company == car.company):  # look for available cars
                    vec1 = [car.pos_x, car.pos_y]
                    vec2 = [d.o_x, d.o_y]
                    car.dist_to_dem = cityblock(vec1, vec2)  # calculate for each car the distance to the chosen demand
            #veh_sorted = sorted(vehicles,
            #                    key=lambda x: x.dist_to_dem)  # sort the vehicles by the distance to the demand
            #selected_car = next((x for x in veh_sorted if x.company == d.company and x.status == 'free'),
            #                    None)   # select the nearest car (the first in sorted list) subject to being free and from the requested company, return None if no car selected
            # if we want to assign the random car for the first demand in the queue:
            selected_car = next((x for x in vehicles if x.company == d.company and x.status == 'free'),
                                None)   # select the nearest car (the first in sorted list) subject to being free and from the requested company, return None if no car selected

            if selected_car is None:    # if no free cars - update the waiting time of demand
                for dem in demands:
                    if dem.id == d.id:
                        dem.wait_match_time += 1
            else:                       # if a car is selected
                for dem in demands:
                    if dem.id == d.id:      # look for this demand in the initial list of demands
                        dem.veh_id = selected_car.id  # assign the id of chosen car to the demand
                        dem.status = 'waiting'  # change the status of demand
                        for car in vehicles:
                            if car.id == selected_car.id:
                                idle_dist_to_queued_dem += car.dist_to_dem
                            car.dist_to_dem = 0  # reset the 'distance to demand' variable
                            if car.id == selected_car.id:
                                car.status = 'goes_to_cust'  # change the status of the car
                                car.dest_x = dem.o_x  # change the car destination coordinates
                                car.dest_y = dem.o_y
                                num_free_veh[car.company] -= 1
                                num_queued_dem_served += 1

        for dem in demands:         # to treat the cases when there was a queue but a lot of cars became free and managed to serve all the queued demand but we are not sure if the new demand will be also served or will go to the queue
            if dem.req_time == time and dem.status == 'on hold':  # choose a demand where the time of receiving the request is equal to the current time and the status of the request is "on hold"
                for car in vehicles:
                    if (car.status == 'free') and (
                            dem.company == car.company):  # look for available cars from the requested company
                        vec1 = [car.pos_x, car.pos_y]
                        vec2 = [dem.o_x, dem.o_y]
                        car.dist_to_dem = cityblock(vec1, vec2)  # calculate for each car the distance to the chosen demand
                #veh_sorted = sorted(vehicles,
                #                    key=lambda x: x.dist_to_dem)  # sort the vehicles by the distance to the demand
                #selected_car = next((x for x in veh_sorted if x.company == dem.company and x.status == 'free'),
                #                    None)  # select the nearest car (the first in sorted list) subject to being free and from the requested company, return None if no car selected
                # if we want to assign the random car for the first demand in the queue:
                selected_car = next((x for x in vehicles if x.company == dem.company and x.status == 'free'),
                                    None)  # select the nearest car (the first in sorted list) subject to being free and from the requested company, return None if no car selected

                if selected_car is None:
                    if dem.wait_match_time < queue_time:   # if no available car - put demand request in a queue for 'queue_time' min
                        if (dem.wait_match_time == 0) and (time >= discarded_time):    # if it's new request - update the number of requests that have been queued
                            num_queued_demands_total[dem.company] += 1

                        dem.status = 'queue'
                        dem.wait_match_time += 1

                    elif dem.wait_match_time >= queue_time:      # if request is in a queue for more than 'queue_time' min - reject it
                        dem.status = 'rejected'
                        if (time >= discarded_time) and (dem.req_time >= discarded_time):
                            num_rejected_demands[dem.company] += 1

                else:
                    dem.veh_id = selected_car.id  # assign the id of chosen car to the demand
                    dem.status = 'waiting'  # change the status of demand
                    for car in vehicles:
                        car.dist_to_dem = 0  # reset the 'distance to demand' variable
                        if car.id == selected_car.id:
                            car.status = 'goes_to_cust'  # change the status of the car
                            car.dest_x = dem.o_x  # change the car destination coordinates
                            car.dest_y = dem.o_y
                            num_free_veh[car.company] -= 1

    return num_queued_demands_total, num_queued_demands_instant, num_rejected_demands, idle_dist_to_queued_dem, num_queued_dem_served


def reaching_customer(discarded_time):
    for car in vehicles:
        if car.status == 'goes_to_cust':
            if car.pos_x == car.dest_x and car.pos_y == car.dest_y:  # if the car have arrived to the customer
                car.status = 'occupied'  # change the status of car
                for dem in demands:
                    if dem.status == 'waiting' and dem.veh_id == car.id:
                        dem.status = 'in car'  # change the status of demand
                        car.dest_x = dem.d_x  # change the destination of car - now it is the destination of customer
                        car.dest_y = dem.d_y

            elif car.pos_x > car.dest_x:  # if not yet arrived to customer - update the location of car every timestep
                car.pos_x -= 1
                for dem in demands:
                    if dem.status == 'waiting' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        dem.idle_dist_dem = + 1
                        car.total_idle_dist += 1

            elif car.pos_x < car.dest_x:
                car.pos_x += 1
                for dem in demands:
                    if dem.status == 'waiting' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        dem.idle_dist_dem = + 1
                        car.total_idle_dist += 1

            elif car.pos_y > car.dest_y:
                car.pos_y -= 1
                for dem in demands:
                    if dem.status == 'waiting' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        dem.idle_dist_dem = + 1
                        car.total_idle_dist += 1

            elif car.pos_y < car.dest_y:
                car.pos_y += 1
                for dem in demands:
                    if dem.status == 'waiting' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        dem.idle_dist_dem = + 1
                        car.total_idle_dist += 1


def reaching_destination(time, discarded_time, num_served_demand):
    for car in vehicles:
        if car.status == 'occupied':
            if car.pos_x == car.dest_x and car.pos_y == car.dest_y:  # if the car have arrived to the destination
                car.status = 'free'  # change the status of car
                for dem in demands:
                    if dem.status == 'in car' and dem.veh_id == car.id:
                        dem.status = 'finished'  # change the status of demand
                        num_free_veh[car.company] += 1
                        if time >= discarded_time:
                            num_served_demand[car.company] += 1

            elif car.pos_x > car.dest_x:  # if not yet arrived to destination - update the location of car every timestep
                car.pos_x -= 1
                for dem in demands:
                    if dem.status == 'in car' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        car.total_occup_dist += 1

            elif car.pos_x < car.dest_x:
                car.pos_x += 1
                for dem in demands:
                    if dem.status == 'in car' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        car.total_occup_dist += 1

            elif car.pos_y > car.dest_y:
                car.pos_y -= 1
                for dem in demands:
                    if dem.status == 'in car' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        car.total_occup_dist += 1

            elif car.pos_y < car.dest_y:
                car.pos_y += 1
                for dem in demands:
                    if dem.status == 'in car' and dem.veh_id == car.id and dem.req_time >= discarded_time:
                        car.total_occup_dist += 1

    return num_served_demand


def avg_distance(num_served_demand):
    idle_dist = {i: 0 for i in companies}
    occup_dist = {i: 0 for i in companies}
    non_moving_time_rate = []

    for car in vehicles:
        idle_dist[car.company] += car.total_idle_dist
        occup_dist[car.company] += car.total_occup_dist


    idle_dist_avg = {i: 0 for i in companies}
    occup_dist_avg = {i: 0 for i in companies}

    for key in companies:
        idle_dist_avg[key] = idle_dist[key] / num_served_demand.get(key)
        occup_dist_avg[key] = occup_dist[key] / num_served_demand.get(key)


    print('idle distance:')
    for key in idle_dist_avg:
        print(key + ' ' + str(idle_dist_avg[key]))

    print('occup distance:')
    for key in occup_dist_avg:
        print(key + ' ' + str(occup_dist_avg[key]))

    return idle_dist_avg, occup_dist_avg

def non_mov_time(max_time, discarded_time):
    non_moving_time = []

    for car in vehicles:
        non_moving_time.append((max_time - discarded_time - car.total_occup_dist - car.total_idle_dist) * 100
                               / (max_time - discarded_time))

    non_moving_time_rate = sum(non_moving_time) / len(non_moving_time)
    print('avg non-moving % of time: ' + str(non_moving_time_rate))
    return non_moving_time_rate


def free_veh_rate(time, discarded_time):
    if time >= discarded_time:
        non_mov_veh_rate.append(num_free_veh['uber'] * 100 / num_veh['uber'])


def st_dev_calcul(demands, demand_count, idle_dist_avg, occup_dist_avg, discarded_time):
    sum_sqrt_idle = 0
    sum_sqrt_occup = 0
    for dem in demands:
        if dem.req_time >= discarded_time:
            sum_sqrt_idle = + (dem.idle_dist_dem - idle_dist_avg['uber']) ** 2
            sum_sqrt_occup = + (dem.occup_dist_dem - occup_dist_avg['uber']) ** 2

    st_dev_idle = math.sqrt(sum_sqrt_idle / demand_count['uber'])
    st_dev_occup = math.sqrt(sum_sqrt_occup / demand_count['uber'])
    return st_dev_idle, st_dev_occup


def printing_cars():
    print('cars:')
    print()
    for i in vehicles:
        print('company: ' + i.company)
        print('id: ' + str(i.id))
        print('pos_x: ' + str(i.pos_x))
        print('pos_y: ' + str(i.pos_y))
        print('status: ' + i.status)
        print('total occupied dist: ' + str(i.total_occup_dist))
        print('total idle dist: ' + str(i.total_idle_dist))
        print()


def printing_demands():
    print('demands:')
    print()
    for i in demands:
        print('company: ' + i.company)
        print('id: ' + str(i.id))
        print('o_x: ' + str(i.o_x))
        print('o_y: ' + str(i.o_y))
        print('d_x: ' + str(i.d_x))
        print('d_y: ' + str(i.d_y))
        print('req_time: ' + str(i.req_time))
        print('status: ' + i.status)
        print('veh id: ' + str(i.veh_id))
        print()


##################################################
### MAIN PART WHERE ALL THE MAGIC IS HAPPENING ###
##################################################

glob_occup_dist_avg = []     #occupied distance run
glob_idle_dist_avg = []      #idle distance run
glob_idle_dist_to_queued_dem_avg = []
glob_demand_count = []
glob_demand_served = []
glob_demand_queued = []     # total # of demand that was queued
glob_demand_rejected = []

glob_demand_queued_avg = []     # avg # of queued demand over time
glob_demand_queued_avg_norm = []

glob_demand_count_norm = []     # normalized demand - density of demand per minute per square km
glob_demand_served_norm = []    # normalized served demand - density of demand per minute per square km
glob_demand_queued_norm = []
glob_demand_rejected_norm = []  # normalized rejected demand - density of demand per minute per square km
glob_demand_rejected_per_veh = []

glob_demand_queued_rate = []
glob_demand_rejection_rate = []
glob_non_moving_time_rate = []
glob_non_mov_veh_rate = []
glob_veh_vacant_dens = []

glob_veh_density = []           # density of total fleet (both vacant and occupied) per square km
glob_request_rate_per_veh = []  # total number of requests taken (not only served)
glob_request_rate_per_veh_per_min = []
glob_request_rate_per_min = []

glob_st_dev_idle = []
glob_st_dev_occup = []


for iter in range(0, 20):                # set the number of simulation runs
    print('_________________________')
    print("ITERATION# " + str(iter))
    max_time = 300  # time horizon
    discarded_time = 100    # time needed to saturate the network. start controlling network after this time
    max_x = 35  # length of x axis of network
    max_y = 35  # length of y axis of network

    queue_time = 200

    companies = ['uber']
    num = random.randrange(10, 1000, 2)  # number of vehicles of each company - normal [200,1000,2]
    #num = 400
    num_veh = {'uber': num}  # number of vehicles of each company
    num_free_veh = {'uber': num}   # number of free (idle non-moving) vehicles
    num_served_demand = {i: 0 for i in companies}
    demand_count = {i: 0 for i in companies}
    num_rejected_demands = {i: 0 for i in companies}
    num_queued_demands_total = {i: 0 for i in companies}
    num_queued_demands_instant = []
    non_mov_veh_rate = []
    idle_dist_to_queued_dem = 0
    num_queued_dem_served = 0

    ### !!! TO ADD NEW COMPANIES - CHANGE THE 'companies' AND 'num_veh' ARRAYS AND THE INITIALIZATION OF VEHICLES

    # initialization of vehicles
    vehicles = initialization_veh(num_veh)

    # initialization of demand
    demands, demand_count = initialization_dem(num_veh, demand_count, discarded_time)

    # printing out cars and demands
    # printing_cars()
    # printing_demands()

    # simulation itself
    for t in range(max_time):
        num_queued_demands_total, num_queued_demands_instant, num_rejected_demands, idle_dist_to_queued_dem, num_queued_dem_served = \
            matching(t, num_queued_demands_total, num_queued_demands_instant, num_rejected_demands, discarded_time, idle_dist_to_queued_dem, num_queued_dem_served)
        reaching_customer(discarded_time)
        num_served_demand = reaching_destination(t, discarded_time, num_served_demand)
        free_veh_rate(t, discarded_time)

    idle_dist_avg, occup_dist_avg = avg_distance(num_served_demand)  # calculating the avg idle and occupied distance
    non_mov_time_rate = non_mov_time(max_time, discarded_time)      # avg % of time when vehicle is non-moving
    st_dev_idle, st_dev_occup = st_dev_calcul(demands, demand_count, idle_dist_avg, occup_dist_avg, discarded_time)
    print()
    print('st dev idle: ' + str(st_dev_idle))
    print('st dev occup: ' + str(st_dev_occup))
    print()
    print('# of vehicles: ' + str(num))

    ####################################################
    ########## Saving info of each iteration ###########
    ####################################################

    ######### Occupied and idle distance

    for key in idle_dist_avg:
        if key == 'uber':
            glob_idle_dist_avg.append(idle_dist_avg[key])

    for key in occup_dist_avg:
        if key == 'uber':
            glob_occup_dist_avg.append(occup_dist_avg[key])

    if num_queued_dem_served != 0:
        glob_idle_dist_to_queued_dem_avg.append(idle_dist_to_queued_dem/num_queued_dem_served)
        print(idle_dist_to_queued_dem)
        print(num_queued_dem_served)
        print("glob_idle_dist_to_queued_dem_avg: " + str(glob_idle_dist_to_queued_dem_avg[-1] * 200))



    ######### Normalizing and saving demands

    for key in demand_count:  # print the quantity of demand of each company
        print('demand: ' + str(demand_count[key]))
        if key == 'uber':
            glob_demand_count.append(demand_count[key])

    for key in demand_count:  # normalized demand - density of demand per minute per square km
        print('demand normalized: ' + str(demand_count[key] / ((max_time - discarded_time) * max_x * max_y)))
        if key == 'uber':
            glob_demand_count_norm.append(demand_count[key] / ((max_time - discarded_time) * max_x * max_y))



    for key in num_served_demand:  # print the quantity of demand of each company
        print('served demand: ' + str(num_served_demand[key]))
        if key == 'uber':
            glob_demand_served.append(num_served_demand[key])

    for key in num_served_demand:   # normalized served demand - density of served demand per minute per square km
        print('served demand normalized: ' + str(num_served_demand[key] / ((max_time - discarded_time) * max_x * max_y)))
        if key == 'uber':
            glob_demand_served_norm.append(num_served_demand[key] / ((max_time - discarded_time) * max_x * max_y))



    ######### Queued demand
    for key in num_queued_demands_total:
        print('queued demand: ' + str(num_queued_demands_total[key]))
        if key == 'uber':
            glob_demand_queued.append(num_queued_demands_total[key])

    for key in num_queued_demands_total:    # normalized queued demand - density of queued demand per minute per square km
        if key == 'uber':
            glob_demand_queued_norm.append(num_queued_demands_total[key] / ((max_time - discarded_time) * max_x * max_y))

    glob_demand_queued_rate.append((num_queued_demands_total['uber'] / demand_count['uber']) * 100)
    print('queued demand %: ' + str((num_queued_demands_total['uber'] / demand_count['uber']) * 100))



    glob_demand_queued_avg.append(sum(num_queued_demands_instant) / len(num_queued_demands_instant))
    print('avg # of queued demands over time:' + str(sum(num_queued_demands_instant) / len(num_queued_demands_instant)))

    glob_demand_queued_avg_norm.append((sum(num_queued_demands_instant) / len(num_queued_demands_instant)) / (max_x * max_y))
    print('avg # of queued demands density:' + str((sum(num_queued_demands_instant) / len(num_queued_demands_instant)) / ((max_time - discarded_time) * max_x * max_y)))


    """on_hold_end = 0
    for d in demands:
        if (d.status == 'on hold') and (d.req_time >= discarded_time):
            on_hold_end += 1
    print('on hold demand at the end: ' + str(on_hold_end))

    waiting_end = 0
    for d in demands:
        if (d.status == 'waiting') and (d.req_time >= discarded_time):
            waiting_end += 1
    print('waiting demand at the end: ' + str(waiting_end))

    queue_end = 0
    for d in demands:
        if (d.status == 'queue') and (d.req_time >= discarded_time):
            queue_end += 1
    print('queued demand at the end: ' + str(queue_end))

    in_car_end = 0
    for d in demands:
        if (d.status == 'in car') and (d.req_time >= discarded_time):
            in_car_end += 1
    print('in car demand at the end: ' + str(in_car_end))

    finished_end = 0
    for d in demands:
        if (d.status == 'finished') and (d.req_time >= discarded_time):
            finished_end += 1
    print('finished demand at the end: ' + str(finished_end))"""


    ######## Rejected demand
    for key in num_rejected_demands:  # print the quantity of demand of each company
        print('rejected demand: ' + str(num_rejected_demands[key]))
        if key == 'uber':
            glob_demand_rejected.append(num_rejected_demands[key])

    for key in num_rejected_demands:    # normalized rejected demand - density of rejected demand per minute per square km
        print('rejected demand normalized: ' + str(num_rejected_demands[key] / ((max_time - discarded_time) * max_x * max_y)))
        if key == 'uber':
            glob_demand_rejected_norm.append(num_rejected_demands[key] / ((max_time - discarded_time) * max_x * max_y))

    for key in num_rejected_demands:    # normalized rejected demand per veh
        if key == 'uber':
            glob_demand_rejected_per_veh.append(num_rejected_demands[key] / num_veh['uber'])

    glob_demand_rejection_rate.append((num_rejected_demands['uber'] / demand_count['uber']) * 100)
    print('rejected demand %: ' + str((num_rejected_demands['uber'] / demand_count['uber']) * 100))

    ################# Vehicle density and request rate per vehicle

    for key in num_veh:         #density of fleet (both vacant and occupied) per square block
        if key == 'uber':
            glob_veh_density.append(num_veh[key] / (max_x * max_y))

    for key in num_veh:         #request rate per vehicle
        if key == 'uber':
            glob_request_rate_per_veh.append(demand_count[key] / num_veh[key])


    for key in num_veh:         #request rate per vehicle per minute
        if key == 'uber':
            glob_request_rate_per_veh_per_min.append(demand_count[key] / (num_veh[key] * (max_time - discarded_time)))


    for key in num_veh:         #request rate  per minute
        if key == 'uber':
            glob_request_rate_per_min.append(demand_count[key] / (max_time - discarded_time))


    ######## Calculating avg non-moving time rate & avg non-moving % of vehicle

    glob_non_moving_time_rate.append(non_mov_time_rate)
    glob_non_mov_veh_rate.append(sum(non_mov_veh_rate) / len(non_mov_veh_rate))
    print('avg non-moving % of vehicles: ' + str(sum(non_mov_veh_rate) / len(non_mov_veh_rate)))
    glob_veh_vacant_dens.append(((sum(non_mov_veh_rate) / len(non_mov_veh_rate)) * glob_veh_density[-1]) / 100)
    print('global veh density: ' + str(glob_veh_density[-1]))
    print('density of vacant vehicles: ' + str(glob_veh_vacant_dens[-1]))


    ################### Standard deviation #########

    glob_st_dev_idle.append(st_dev_idle)
    glob_st_dev_occup.append(st_dev_occup)

print()
print("avg occup distance: " + str(sum(glob_occup_dist_avg) / len(glob_occup_dist_avg)))
print("avg idle distance: " + str(sum(glob_idle_dist_avg) / len(glob_idle_dist_avg)))
glob_st_dev_occup_avg = sum(glob_st_dev_occup) / len(glob_st_dev_occup)
print("avg sigma occup: " + str(glob_st_dev_occup_avg))
print()
print('Total demand:')
print(glob_demand_count)

#print('Queued demand:')
#print(glob_demand_queued)
#print('Queued demand rate, %:')
#print(glob_demand_queued_rate)
#print('Avg demand queued rate, %:')
#print(sum(glob_demand_queued_rate) / len(glob_demand_queued_rate))

#print('Rejected demand:')
#print(glob_demand_rejected)
#print('Rejected demand rate, %:')
#print(glob_demand_rejection_rate)
#print('Avg demand rejection rate, %:')
#print(sum(glob_demand_rejection_rate) / len(glob_demand_rejection_rate))
print('Request rate per veh:')
print(glob_request_rate_per_veh)
"""
queued_idle_dist = sum(glob_idle_dist_to_queued_dem_avg)/len(glob_idle_dist_to_queued_dem_avg)
print("Average idle dist to queued dem: " + str(sum(glob_idle_dist_to_queued_dem_avg)/len(glob_idle_dist_to_queued_dem_avg)))
sd = 0
for i in range(0, len(glob_idle_dist_to_queued_dem_avg)):
    sd += (glob_idle_dist_to_queued_dem_avg[i] - sum(glob_idle_dist_to_queued_dem_avg)/len(glob_idle_dist_to_queued_dem_avg)) ** 2
st_dev_queued_idle_dist = math.sqrt(sd/len(glob_idle_dist_to_queued_dem_avg))
print("St dev of the average idle dist to queued dem: " + str(st_dev_queued_idle_dist))"""

dataframe_dict = {
            'glob_occup_dist_avg': glob_occup_dist_avg,
            'glob_idle_dist_avg': glob_idle_dist_avg,
            'glob_demand_count': glob_demand_count,
            'glob_demand_served': glob_demand_served,
            #'glob_demand_queued': glob_demand_queued,
            'glob_demand_rejected': glob_demand_rejected,
            'glob_demand_count_norm': glob_demand_count_norm,
            'glob_demand_served_norm': glob_demand_served_norm,
            #'glob_demand_queued_norm': glob_demand_queued_norm,
            'glob_demand_rejected_norm': glob_demand_rejected_norm,
            'glob_demand_rejected_per_veh': glob_demand_rejected_per_veh,
            #'glob_demand_queued_rate': glob_demand_queued_rate,
            #'glob_demand_queued_avg': glob_demand_queued_avg,
            #'glob_demand_queued_avg_norm': glob_demand_queued_avg_norm,     #### !!!!
            'glob_demand_rejection_rate': glob_demand_rejection_rate,
            'glob_non_moving_time_rate': glob_non_moving_time_rate,
            'glob_non_mov_veh_rate': glob_non_mov_veh_rate,
            'glob_veh_vacant_dens': glob_veh_vacant_dens,
            'glob_veh_density': glob_veh_density,
            'glob_request_rate_per_veh': glob_request_rate_per_veh,
            'glob_request_rate_per_veh_per_min': glob_request_rate_per_veh_per_min,
            'glob_request_rate_per_min': glob_request_rate_per_min,
            'glob_st_dev_idle': glob_st_dev_idle,
            'glob_st_dev_occup': glob_st_dev_occup
            #'glob_idle_dist_to_queued_dem_avg': glob_idle_dist_to_queued_dem_avg
          }

df = pd.DataFrame(dataframe_dict)
df.to_csv('/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/microsim_result_data/big_num_of_veh.csv')


"""
dataframe_dict2 = {
            'queued_idle_dist': [queued_idle_dist],
            'st_dev_queued_idle_dist': [st_dev_queued_idle_dist]
          }
"""
#df2 = pd.DataFrame(dataframe_dict2)
#df2.to_csv('/Users/maryia/PycharmProjects/SimulationTrial/micro_sim/queued_idle_dist.csv')