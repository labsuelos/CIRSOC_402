Developer's corner
==================

Welcome, my weary engineering adventurer! Please, sit down by the fire
and rest of your daily toil.

I gather that you are here because you have become interested on how the
code behind ``cirsoc_402`` works beyond the point of just using it. You
should know then that ``cirsoc_402`` has two sets of intedend users:
people who want to design foundations and people who want to develop
engineering standards on how to design foundations. If you are reading
this for the first time, then you have been in the first group so far.
What set them appart? Well, to put it bluntly, craziness. The first
group is the average and well-adjusted car owner. They want a reliable
machine that will take them from A to B without much trouble. They
really don't care much about the machine except when it brakes. If it
breaks they take it to someone who will touch all the parts and pieces
they dare not touch for fear of the engine blowing apart. And if the
engine does explode they get to blame the mechanic. The second kind,
those poor souls, are crazy car enthusiast that find pleasure in taking
apart they car and built it over and over again. Unfortunately for them,
half ot the time they are driving happily when they realize that the
front left wheel has decided it was a good time to perform an
unscheduled rapid disassembly.

These two groups interact with ``cirsoc_402`` in very different ways.
Consquently, the program expects very different inputs and skills from
them. The design users deal mostly with a set of predifined objects
such as ``Shallow`` or ``Pile``. These objects perform tons of input
validation before runing a single computation and if they see something
they don't like, they stop mid excecution and let you know it. They are
reliable, but ill temperated beasts. This makes sense in a setting where
you want to obtain a ”good” result and you need to make sure that
everything is done properly.

The developers crowd interact with the stuff that is runing in the
background of ``Shallow`` or ``Pile`` and actually doing the computations.
These are a bunch of functions that will plow through whatever data you
put into them unapologetically. They are meant to run, deliver
some output and carry on. If what they deliver is completely useless
because you loaded them with trashy inputs, it's your problem. This is
done so they can go through large data sets without stoping every time
something is not perfectly right. In an ideal world all data would
be perfectly indexed and complete. You don't get to live there. These
functions deal with the gritty reality. The objective is to analyze as
many cases as possible as fast as possible. When something goes wrong
they limit themselves to returning a *not a number* output. When, where,
and, for the love of everything that is good and fair, why they decided
to that is probably going to consume the rest of your day. I hope you
didn't had any plans, or a life.

To be clear, in both cases the ultimate resposability on the results
lies in the user and not ``cirsoc_402``. Even when something can be
potentially wrong in ``cirsoc_402``. The difference lies in how these
functions deal with their inputs, and how much you need to know to
identify the issue and fix it. Here lies the trade-off. You can be very
inflexible with the inputs and produce good quality results in a
reliable way. But this comes at the cost of a very narrow scope. For the
vast majority fo design cases this is exactly what you need. But, when
you are trying to work out the fringes, you need to be able to handle
unexpected stuff and keep going.

.. toctree::
   :maxdepth: 1
   
   Loads and reference systems <developers/loads.ipynb>
