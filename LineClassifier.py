"""
A class that represents a line classifier, y=mx+b
Has the following attributes: m (the slope), b (the y-intercept), and a boolean called "up" that says if
the line classifies all the points above it as positive (and below it as negative), or not.
"""


class LineClassifier:

    def __init__(self, x1, y1, x2, y2, up=True):  # the default classification is 1 above the line and -1 below it
        self.up = up
        if x1 != x2:
            self.m = (y2-y1) / (x2-x1)  # calculate m
            self.b = y1 - self.m*x1  # find b after we have m
            self.indicator = 0  # indicator to the event "line parallel to y axis"
        else:
            self.indicator = 1
            self.x = x1

    def get_classification(self, x, y):
        if self.indicator == 0:
            result = self.m*x + self.b  # compute the result of line given x
            if result >= y:  # the point (x,y) is above the line
                if self.up:
                    return 1
                else:
                    return -1
            else:  # the point (x,y) is below the line
                if self.up:
                    return -1
                else:
                    return 1
        else:
            if x > self.x:
                if self.up:
                    return 1
                else:
                    return -1
            else: 
                if self.up:
                    return -1
                else:
                    return 1 
