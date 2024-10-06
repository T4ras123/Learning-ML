def eucledian_distance(x, y):
    """_summary_

    Args:
        x (array): first point in n-dimentional space
        y (array): second point in n-dimentional space 

    Returns:
        float: distance between two points
    """
    dist = 0
    
    for n in range(len(x)):
        dist += (x[n] - y[n])**2
        
    return dist**0.5

def KNN(points, p, k=3):
    """
    This function finds the classification of p using
    k nearest neighbor algorithm. It assumes only two
    groups and returns 0 if p belongs to group 0, else
    1 (belongs to group 1).
    
    Args:
        points (dictionary): training points with lables 0 or 1 
        p (tuuple): point to classify
        k (int, optional): number of nearest neighbors. Defaults to 3.
    """
    
    distance = []
    for group in points:
        for feature in points[group]:
            dist = eucledian_distance(feature, p)
            distance.append((dist, group))
    
    distance = sorted(distance)[:k]
    freq1 = 0
    freq2 = 0
    
    for i in distance:
        if i[1] == 0: freq1+=1
        else: freq2 +=1
        
    return 0 if freq1>freq2 else 1


def main():
    
    points = {0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)],
              1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}
    
    p = (2.5 ,7)
    
    k = 3
    
    print("The value classified to unknown point is: {}".\
          format(KNN(points,p,k)))

if __name__ == '__main__':
    main()