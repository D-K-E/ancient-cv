#########################
Circular Hough Transform
#########################

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

So in order to determine how many points are on the same line we need to observe
how many points have same coordinate, in hough space. Each group of points
that have the same hough space coordinate, belongs to a single line.

So the algorithm is the following:

- We detect the edges of the image, no relation to hough transform
- Obtain the coordinates of the edges.
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

There are complications with this approach but it sums up well the basic
strategy. For more information on line detection with hough transform,
see line detection with hough transform.

Circle Detection and Hough Transform
======================================

Now the idea is very similar for circle detection hough transform.
We transform the circle to a parameter space, where each circle is represented
as a x, y, z coordinate.

Let's remember what a circle is.
A circle in the cartesian coordinate system is characterized by the following
equation: :math:`r^2 = (c-a)^2 + (d-b)^2`
where

- r is the radius
- a is the coordinate of the center in x direction
- b is the coordinate of the center in y direction
- c is the coordinate of an arbitrary point on the circle in x direction
- d is the coordinate of an arbitrary point on the circle in y direction

The math behind this is simple pythagoras theorem.
We can visualize this as the following.

I choose an arbitrary point on the circle.
From that point I draw a line parallel to the y axis towards the direction of
the center.

I draw a line from the center to the newly drawn parallel line.
They intersect and the angle they make is 90 degrees, since the line is drawn
parallel to y axis and the line coming from the center is parallel to the x
axis. 

This way the radius becomes the hypothenus of a right triangle where one side
is the difference between the center and the point in x direction, and the
other side in y direction

Thus if we know the coordinates of the center of the circle, and the radius
we can check easily whether a point lies on the circumference of the circle
or not.

So if we want to map the circle to a parameter space, the axis for the parameter
space would be x, y for mapping the coordinates of the center of the circle and
z for mapping the radius of the circle. Hence the circle is parametrized in 3d
space.

A circle's circumference can be computed as
:math:`2 {\times} \pi {\times} radius`
Its area can be computed as :math:`\pi {\times} r^2`.

The general algorithm of hough transform does not really change:

We consider each edge coordinate as a possible center and collect the number
of points that falls to the circumference of the circle.

Then we select from the gathered the coordinates that gathered the maximum
number of points around its circumference.

If we don't know exactly the radius we simply give a range
