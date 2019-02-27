#####################################
Line Detection with Hough Transform
#####################################

Hough Transform and Line Detection
===================================

In general hough transform means we represent a given shape in xy coordinate
plane differently.

For example a line in xy coordinate plane is y = mx + b where x, y are
coordinate points, m is the slope of the line and b is the intercept that is the
point where the line intercepts the y axis, ie the y value if the x=0.

The line can also be represented as a function on m,b space,
Ex. F(m,b) = mx + b - y
since that which distinguishes a line in coordinate space is the slope and
the intercept of a line, we can also represent it in the slope, intercept space,
by a single point (m,b). 

Now the idea is following for hough transform. Let's say you have several
isolated points in coordinate space, if you want to know whether they are on
the same line, you can simply transform their coordinates to hough space, ie
m, b space and see if they give the same result.
Because if they are on the same line, they should have the same, m,b
thus should have the same coordinates on Hough space.

Here we should consider the following: What if we have a line that is parallel
to the y axis. This would force us to have an infinite size for the parameter
space. For computing purposes, we don't want that.
It is better to represent the line with its norm. A norm of the line A is an
intersection point between A and the line B that goes through the origin and
the norm point.

The relationship between A and B is so that B is perpendicular to A.
Since there can only be a one point on the line A, that satisfies this condition
we can also represent the line A, with that point.
So let's say x, y are on the line A and B.
We know that B passes through the origin of the coordinate system (o_x, o_y).
We can thus say that the distance between the origin and the point on A,
is x^2 + y^2 = distance^2 using pythogorian theorem.
Let's also imagine that we draw a perpendicular line from the norm point to the
x axis.

This would create a right angle triangle whose hypothenus is the distance
between the origin and the norm point.
x and y can also be expressed with trigonometric functions of the angle that
the distance makes with the origin. 

Notably sinus of the angle would be (y - o_y) / distance,
and the cosinus of the angle would be (x - o_x) / distance

Suppose for the sake of simplifying the calculations the origin is
(0,0), so the (o_x, o_y) = (0, 0)

This has implications for the x notably x = (x - o_x) becomes
x = (x - 0) so simply x. Nonethless the idea laid out here would have worked
the same without having (0,0)
so x can be rewritten as x = distance {\times} cos(theta)
y can also be rewritten as y = distance {\times} sin(theta)

so the point (x,y) in the cartesian coordinate system, can be considered as
distance {\times} ( cos(theta), sin(theta) ), meaning that the vector
represented by (x,y) can be considered as a scaled by distance version of
the vector ( cos(theta), sin(theta) ) which is represented with a point on the
unit circle.

so simply put the only difference between
the point (x, y) and (cos(theta), sin(theta) ) is the magnitude of the vector.
This magnitude is the distance from the origin.

The nice thing about using theta and the distance for representing points, is
the fact that the parameter space becomes finite.

Theta is a number between 0 and 2pi, and the distance is at most the size
of the diagonal of the image

Now in order to write the equation of line A in polar coordinates, you need the
following:
( this is very well explained in the
video[https://www.youtube.com/watch?v=SifX0ycUCVs]_ )

- Point C on the line A that is closest to the origin O.
- Another arbitrary point B on the line A.
- The angle T of the line D that passes through the origin O and point C
- The angle P of the line Q that passes through the origin O and point B

Now the line D makes a right angle with the line A, since the closest point
to the origin on the line A lies on straight line coming from the origin which
intersects the line A

With the right angle on A we can use trigonometry.
The angle between line D and Q is (P - Q)
Let's say the distance between the origin and point B is k and the distance
between the origin and point C is z.

k is the hypothenus of the right triangle, and z is one of its sides.
The angle between them is (P-Q).
So basically, cos(P-Q) = z/k
thus k {\times} cos(P-Q) = z
and k = z / cos(P-Q)
This equation k = z / cos(P-Q) caracterizes the line A with polar coordinates

So the algorithm is the following:

- We detect the edges of the image, no relation to hough transform
- Obtain the coordinates of the edges.

Here we have 2 choices,
if we choose to use m, b for our parameter space we do the following:
So in order to determine how many points are on the same line we need to observe
how many points have same coordinate, in hough space. Each group of points
that have the same hough space coordinate, belongs to a single line.

- Create an accumulator matrix, that would contain two following information:
  - A hough space coordinate,
  - Number of pixels in the image that are transformed to above mentioned
    hough space coordinate, basically a counter
- For each coordinate of a bright pixel included in the edges, do:
  - Transform the pixel coordinate to hough space
  - if the hough space coordinate already exist within the accumulator
    matrix/hash table, increment its counter.
  - If the hough space coordinate is not present in the accumulator:
    - Add it to the accumulator.

If we choose polar coordinates as our choice we do the following:

- We transform all edge coordinates to polar coordinates
- Create an accumulator hash table that contains the following:
  - Polar coordinates of edges that satisfies the equation
  - number of points that satisfies the equation

- After this we can select the groups that are below or above a certain number
- And convert them back to their original coordinates
