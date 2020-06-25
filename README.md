A quick project I did to learn nom and also because it's a program I find genuinely useful.
When I lived out west, I would use the time-height cross-sections at weather.utah.edu very often.
Now that I live in the east, I missed that, so I wrote something that would reproduce those 
cross-sections (though not nearly as prettily).

I used nom to parse the (undocumented) BufKit file format. That could be extracted into its own 
crate if anyone is interested, but I didn't want to make a coherent API for that for a personal 
project.
