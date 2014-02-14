import pickle

import urllib
from HTMLParser import HTMLParser
from htmlentitydefs import name2codepoint
import sys
import os
import tornado.httpclient

actors = []

def DumpPickle(data, path):
  f = open(path, 'wb' )
  pickle.dump(data, f)

def LoadPickle(path):
  f = open(path, 'rb' )
  entry = pickle.load(f)
  return entry

def WriteStrToFile(string, path):
  o = open(path, 'wb')
  o.write(string)
  o.close()
  
class VideoHTMLParser(HTMLParser):
        detect = False
	def handle_starttag(self, tag, attrs):	
		if tag == 'span':
			attr_t2d = {}
			for attr in attrs:
				attr_t2d[attr[0]] = attr[1]
			if 'class' in attr_t2d and 'itemprop' in attr_t2d and attr_t2d['class'] == 'itemprop' and attr_t2d['itemprop'] == 'name': 
                                VideoHTMLParser.detect = True
	def handle_endtag(self, tag):
		pass
	def handle_data(self, data):
                if VideoHTMLParser.detect:
                    actors.append(data)
                    VideoHTMLParser.detect = False

def GetActor(url_path, dic_path, actor_dic):
    error_path = 'error_url'
    error_url = LoadPickle(error_path)
    http = tornado.httpclient.HTTPClient()
    fh = open(url_path)
    num = 0 
    count = 0
    for url in fh:
      num += 1
      url = url.strip()
      print 'num:' + str(num) + ':' + url
      if num == 253:
        continue
      if num == 525:
        continue
      if num == 528:
        continue
      if num == 886:
        continue
      if num == 889:
        continue
      if num == 977:
        continue
      if num == 1484:
        continue
      if num == 1483:
        continue
      if num == 1485:
        continue
      if num == 1531:
        continue
      if num == 1533:
        continue
      if num < 1520:
        continue
      if num not in actor_dic and url.startswith('http'):
        try:
            response = http.fetch(url)
            page = response.body

            pos1 = page.rfind("See full cast")
            pos2 = page.find("<a href=",pos1-100)

            if pos1 < 0 :
                error_url[num] = url
                print str(num) + ':' + url
                DumpPickle(error_url, error_path)
                continue
            
            tmp = "fullcredits?ref_=tt_cl_sm#cast"
            flevel_url = response.effective_url + tmp

            response = http.fetch(flevel_url)
            parser = VideoHTMLParser()
            parser.feed(response.body)

            print num
            actor_dic[num] = '|'.join(actors)
            count += 1
            if count % 5 == 0:
              DumpPickle(actor_dic, dic_path)
            del actors[:]
            print len(actor_dic)

        except tornado.httpclient.HTTPError as e:
            print "Error:", e

def InitDic(path):
  actor_dic = {}
  DumpPickle(actor_dic, path)

def Filter(actor_dic, num):
  actor_dic2 = {}
  for i in actor_dic:
    if len(actor_dic[i]) > num:
      actor_dic2[i] = actor_dic[i]
  return actor_dic2

def MeanValueSize(dic):
  s = 0.0
  for i in dic:
    s += len(dic[i])
  return s / len(dic)

def GetActorDic(title_dic, num):
  actor_dic = {}
  for t in title_dic:
    l = title_dic[t]
    for e in l:
      if e in actor_dic:
        actor_dic[e.strip()].append(t)
      else:
        actor_dic[e.strip()] = []
        actor_dic[e.strip()].append(t)
  actor_dic = Filter(actor_dic, num)
  print MeanValueSize(actor_dic)
  return actor_dic

def GetActorId(actor_dic):
  id_dic = {}
  i = 1
  for t in actor_dic:
    id_dic[t] = i
    i += 1
  return id_dic

def AddDic(title_dic, title, id):
  if title not in title_dic:
    title_dic[title] = []
  title_dic[title].append(id)

def Unique(data):
  dic = {}
  for d in data:
    dic[d] = ''
  return dic.keys()

def UniqueDic(dic):
  for d in dic:
    dic[d] = Unique(dic[d])
  return dic

def Index(title_dic, min_num):
  for t in title_dic:
    l = (title_dic[t]).split('|')
    title_dic[t] = Unique(l)
  actor_dic = GetActorDic(title_dic, min_num)
  id_dic = GetActorId(actor_dic)
  title_dic2 = {}
  for t in title_dic:
    actor = title_dic[t]
    for a in actor:
      if a in id_dic:
        AddDic(title_dic2, t, id_dic[a])
  return title_dic2

def SaveDoc(path, title_dic):
  key = title_dic.keys()
  key.sort()
  f = open(path, 'wb')
  for k in key:
    a = [] 
    a.append(str(len(title_dic[k])))
    actor = title_dic[k]
    actor.sort()
    for x in actor:
      a.append(str(x) + ':' + '1')
    f.write(' '.join(a))
    f.write('\n')
  f.close()

def CompressedId(dic):
  dic2 = {}
  k = dic.keys()
  k.sort()
  i = 0
  for kk in k:
    dic2[kk] = i
    i += 1
  return dic2

def ConvertRating(src_path, des_path, dic):
  dic2 = CompressedId(dic)
  ll = open(src_path, 'rb')
  result = []
  for l in ll:
    r = l.split(' ')
    if int(r[1]) in dic2:
      r[1] = str(dic2[int(r[1])])
      result.append(' '.join(r))
  ll.close()
  ll = open(des_path, 'wb')
  ll.write(''.join(result))
  ll.close()
      
def main():
  root_path = '/home/lijk/working/project/git/project/trunk/'
  path = './url'
  dic_path = './url_actor'
  #InitDic(dic_path)
  title_dic = LoadPickle(dic_path)
  print len(title_dic)
  print 'load'
  #GetActor(path, dic_path, title_dic)
  min_actor_num = 3 #min number of movie by an actor 
  title_dic = Index(title_dic, min_actor_num)
  print len(title_dic)
  #print MeanValueSize(title_dic)
  doc_path = os.path.join(root_path, 'ml/rbm/tmp/fengxing/data/movie_doc')
  SaveDoc(doc_path, title_dic)
  src_path = os.path.join(root_path, 'data/movielen/movielen_train2.txt')
  des_path = os.path.join(root_path, 'data/movielen/movielen_train3.txt')
  print src_path
  ConvertRating(src_path, des_path, title_dic)
  src_path = os.path.join(root_path, 'data/movielen/movielen_test2.txt')
  des_path = os.path.join(root_path, 'data/movielen/movielen_test3.txt')
  ConvertRating(src_path, des_path, title_dic)

main()
