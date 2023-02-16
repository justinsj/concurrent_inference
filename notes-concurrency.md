# GPU
(concurrency) jd2sanju@gpu1:~/concurrent_inference$ python count_objects.py -f input_folder -o vit-f.log -q 1000 -r 1000 -m vit_h14_in1k -s 518 -d 1

processing 100 images... 
time taken : 76.48491072654724 s.
rate : 13.074474304810932 images/s
{0: {'rate': 1.4406614499313004, 'duration': 69.41256046295166, 'count': 100}}

(concurrency) jd2sanju@gpu1:~/concurrent_inference$ python count_objects.py -f input_folder -o vit-f.log -q 1000 -r 1000 -m vit_h14_in1k -s 518 -d 2
processing 100 images... 
time taken : 67.85863971710205 s.
rate : 14.736517032597918 images/s
{0: {'rate': 0.8373188639149987, 'duration': 59.71440768241882, 'count': 50}, 1: {'rate': 0.8241074212693656, 'duration': 60.67170214653015, 'count': 50}}

(concurrency) jd2sanju@gpu1:~/concurrent_inference$ python count_objects.py -f input_folder -o vit-f.log -q 1000 -r 1000 -m vit_h14_in1k -s 518 -d 4
processing 100 images... 
time taken : 71.3602979183197 s.
rate : 14.01339441077752 images/s
{1: {'rate': 0.3875653158618583, 'duration': 61.92504596710205, 'count': 24}, 2: {'rate': 0.3866147079043312, 'duration': 62.07730722427368, 'count': 24}, 3: {'rate': 0.3984472316342515, 'duration': 62.74356555938721, 'count': 25}, 0: {'rate': 0.4251059996481018, 'duration': 63.51357078552246, 'count': 27}}

(concurrency) jd2sanju@gpu1:~/concurrent_inference$ python count_objects.py -f input_folder -o vit-f.log -q 1000 -r 1000 -m vit_h14_in1k_half -s 518 -d 1
processing 100 images... 
time taken : 37.45198702812195 s.
rate : 26.700852994772212 images/s
{0: {'rate': 3.558187982684774, 'duration': 28.10419249534607, 'count': 100}}

(concurrency) jd2sanju@gpu1:~/concurrent_inference$ python count_objects.py -f input_folder -o vit-f.log -q 1000 -r 1000 -m vit_h14_in1k_half -s 518 -d 2
processing 100 images... 
time taken : 37.83119606971741 s.
rate : 26.43321131473467 images/s
{0: {'rate': 1.607408864921353, 'duration': 30.483843326568604, 'count': 49}, 1: {'rate': 1.650441943484882, 'duration': 30.900814294815063, 'count': 51}}

(concurrency) jd2sanju@gpu1:~/concurrent_inference$ python count_objects.py -f input_folder -o vit-f.log -q 1000 -r 1000 -m vit_h14_in1k_half -s 518 -d 4
processing 100 images... 
time taken : 39.13815093040466 s.
rate : 25.55051723772533 images/s
{1: {'rate': 0.8552145346822886, 'duration': 30.40172839164734, 'count': 26}, 2: {'rate': 0.7845278834066072, 'duration': 30.591646909713745, 'count': 24}, 3: {'rate': 0.7677345273438811, 'duration': 31.26080584526062, 'count': 24}, 0: {'rate': 0.8269148133147476, 'duration': 31.442174673080444, 'count': 26}}





# CPU
# d  = 2
time taken : 352.5613965988159 s.
rate : 2.8363854059096343 images/s
{1: {'rate': 0.1448503137562622, 'duration': 345.18392610549927, 'count': 50}, 0: {'rate': 0.14466286685071017, 'duration': 345.6311981678009, 'count': 50}}

# d = 1
time taken : 471.1740267276764 s.
rate : 2.1223580742449717 images/s
{0: {'rate': 0.21551403905041194, 'duration': 464.00689458847046, 'count': 100}}


# Half model
# d = 1
time taken : 228.95443177223206 s.
rate : 4.3676813427871 images/s
{0: {'rate': 0.4519541778316436, 'duration': 221.26136875152588, 'count': 100}}

# d = 2
time taken : 177.47269415855408 s.
rate : 5.634669630397339 images/s
{0: {'rate': 0.29299050104367963, 'duration': 170.6539967060089, 'count': 50}, 1: {'rate': 0.2929742170291948, 'duration': 170.6634819507599, 'count': 50}}

# d = 4
time taken : 401.7859184741974 s.
rate : 2.4888876240301085 images/s
{2: {'rate': 0.06385001170833611, 'duration': 391.54260635375977, 'count': 25}, 1: {'rate': 0.06345192624149264, 'duration': 393.9990711212158, 'count': 25}, 0: {'rate': 0.06344347218913102, 'duration': 394.0515727996826, 'count': 25}, 3: {'rate': 0.06335218738226772, 'duration': 394.6193656921387, 'count': 25}}