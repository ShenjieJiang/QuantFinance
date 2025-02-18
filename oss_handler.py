import import_ipynb
import oss2
import sys
from setting import SETTINGS
import pickle
import pandas as pd
import os
import io
from dateutil import parser
import datetime
import pandas as pd
from fastparquet import write
from pathlib import  Path

AccessKeyId = SETTINGS["oss.accesskey"]
AccessKeySecret = SETTINGS["oss.secret"]

BucketName = SETTINGS["oss.bucketname"]
Endpoint = SETTINGS["oss.endpoint"]


class newBytes(io.BytesIO):
    def close(self):
        pass


class OssClient(object):
    __instance = None
    __first_init = False

    # 单例模式
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        cls = self.__class__
        if not cls.__first_init:
            self.auth = oss2.Auth(AccessKeyId, AccessKeySecret)
            self.bucket = oss2.Bucket(self.auth, Endpoint, BucketName)
            cls.__first_init = True


    def upload_file_from_fileobj(self, object_name, local_file_path):
        """
            upload_file_from_fileobj方法：上传文件对象到oss存储空间, 该方法可用于我们从上游服务接收了图片参数，然后以二进制形式读文件，上传到oss存储空间指定位置（abc/efg/00），
        当然也可以将本地文件上传到oss我们的bucket. 其中fileobj不止可以是文件对象，也可以是本地文件路径。 put_object方法底层仍是RESTful API的调用，可以指定headers，规定Content-Type等内容
        """
        # 判断bucket中文件是否存在，也可以不判断，会上传更新
        #exist = self.bucket.object_exists(object_name) #<yourObjectName>
        #if exist:
        #    return True
        with open(local_file_path, 'rb') as fileobj:
            result = self.bucket.put_object(object_name, fileobj) #<yourObjectName>
        if result.status == 200:
            return True
        else:
            return False

    def upload_pickle_data(self,df, target_path, *args, report_date=None):
        if isinstance(report_date, datetime.date):
            d = report_date.strftime("%Y-%m-%d")
        if isinstance(report_date, str):
            d = parser.parse(report_date).strftime("%Y-%m-%d")
        if args:
            d = [arg for arg in args][0]
        pickle_buffer = io.BytesIO()
        pickle.dump(df, pickle_buffer)
        target_file_key = os.path.join(target_path, '{}.pkl'.format(d)).replace("\\","/")
        result = self.bucket.put_object(target_file_key, pickle_buffer.getvalue())
        if result.status == 200:
            return True
        else:
            return False


    def list_files(self,prefix = None):
        res = []
        for object_info in oss2.ObjectIterator(self.bucket,prefix):
            print(object_info.key)
            res.append(object_info.key)
        return res

    def upload_parquet_data(self,df, target_path, *args, report_date=None):
        if isinstance(report_date, datetime.date):
            d = report_date.strftime("%Y-%m-%d")
        if isinstance(report_date, str):
            d = parser.parse(report_date).strftime("%Y-%m-%d")
        if args:
            d = [arg for arg in args][0]
        target_file_key = os.path.join(target_path, '{}.parquet'.format(d)).replace("\\","/")
        mem_buffer = newBytes()
        df.to_parquet('noname', engine='fastparquet', open_with=lambda x, y: mem_buffer)
        result = self.bucket.put_object(target_file_key, mem_buffer.getvalue())
        #f = Path(os.getcwd())/'tmp.parquet'
        #write(f, df)
        #with open(f, 'rb') as fileobj:
        #    result = self.bucket.put_object(target_file_key, fileobj)
        #os.remove(f)
        if result.status == 200:
            return True
        else:
            return False

    def save_data_to_pickle(self,df, file_dir_path, *args,report_date=None):
        if isinstance(report_date, datetime.date):
            d = report_date.strftime("%Y-%m-%d")
        if isinstance(report_date, str):
            d = parser.parse(report_date).strftime("%Y-%m-%d")
        if args:
            d = [arg for arg in args][0]
        target_file_key = os.path.join(file_dir_path, '{}.pkl'.format(d))
        with open(target_file_key, 'wb') as f:
            pickle.dump(df, f)

    def save_data_to_parquet(self,df, file_dir_path, *args,report_date=None):
        if isinstance(report_date, datetime.date):
            d = report_date.strftime("%Y-%m-%d")
        if isinstance(report_date, str):
            d = parser.parse(report_date).strftime("%Y-%m-%d")
        if args:
            d = [arg for arg in args][0]
        target_file_key = os.path.join(file_dir_path, f'{d}.parquet')
        df.to_parquet(target_file_key)


    def read_oss_pickle_file(self,object_name):
        """
            download_file_to_fileobj：下载文件到文件流对象。由于get_object接口返回的是一个stream流，需要执行read()后才能计算出返回Object数据的CRC checksum，因此需要在调用该接口后做CRC校验。
        """
        object_stream = self.bucket.get_object(object_name) #<yourObjectName>
        result = object_stream.read()
        if object_stream.client_crc != object_stream.server_crc:
            print("The CRC checksum between client and server is inconsistent!")
            result = None
        return pickle.loads(result)

    def read_oss_parquet_file(self,object_name):
        """
            download_file_to_fileobj：下载文件到文件流对象。由于get_object接口返回的是一个stream流，需要执行read()后才能计算出返回Object数据的CRC checksum，因此需要在调用该接口后做CRC校验。
        """
        object_stream = self.bucket.get_object(object_name) #<yourObjectName>
        result = object_stream.read()
        i = io.BytesIO(result)
        if object_stream.client_crc != object_stream.server_crc:
            print("The CRC checksum between client and server is inconsistent!")
            result = None
        return pd.read_parquet(i)


    def download_file_to_loaclfilepath(self, object_name, local_file_path):
        """
            download_file_to_loaclfilepath：下载文件到本地路径。get_object和get_object_to_file的区别是前者是获取文件流实例，可用于代码处理和远程调用参赛。后者是存储到本地路径，返回的是一个http状态的json结果
        """
        result = self.bucket.get_object_to_file(object_name, local_file_path) # ('<yourObjectName>', '<yourLocalFile>')
        if result.status == 200:
            return True
        else:
            return False

    def generate_temporary_download_url(self,object_name):
        """
            generate_temporary_download_url: 生成加签的临时URL以供授信用户下载。一般在实际业务中，我们是提供给调用方一个临时下载链接，来让其获取文件数据，而不是直接使用以上暴露AccessKeyId和AccessKeySecret的方法。
            因此一般我们会存储某条数据oss的路径（<yourObjectName>）与调用方某个唯一标识的对应关系（如手机号身份证号），在调用方请求时，通过该标识获取其数据的oss文件路径（<yourObjectName>），
            然后制定过期时间，为其生成临时下载链接
            http://bucketname.oss-ap-south-1.aliyuncs.com/abc/efg/0?OSSAccessKeyId=LTA************oN9&Expires=1604638842&Signature=tPgvWz*************Uk%3D
        """
        res_temporary_url = self.bucket.sign_url('GET', object_name, 60, slash_safe=True)
        return res_temporary_url


oss_client = OssClient()