import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from service import hr_service as hr
from entity import organizations as o
from service import recognize_service as rs
import cv2
import numpy as np

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # 数据文件夹根目录
PIC_PATH = os.path.join(PATH , "faces")  # 照片文件夹
DATA_FILE = os.path.join(PATH , "employee_data.txt" ) # 员工信息文件
WORK_TIME = os.path.join(PATH , "work_time.txt" ) # 上下班时间配置文件
USER_PASSWORD = os.path.join(PATH , "user_password.txt" ) # 管理员账号密码文件
RECORD_FILE = os.path.join(PATH , "lock_record.txt" ) # 打卡记录文件
IMG_WIDTH = 640  # 图像的统一宽度
IMG_HEIGHT = 480  # 图像的统一高度


# 自检，检查默认文件缺失
def checking_data_files():
    if not os.path.exists(PATH):
        os.mkdir(PATH)
        print("数据文件夹丢失，已重新创建：" + PATH)
    if not os.path.exists(PIC_PATH):
        os.mkdir(PIC_PATH)
        print("照片文件夹丢失，已重新创建：" + PIC_PATH)
    sample1 = os.path.join(PIC_PATH , "1000000000.png")  # 样本1文件路径
    if not os.path.exists(sample1):
        sample_img_1 = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)  # 创建一个空内容图像
        sample_img_1[:, :, 0] = 255  # 改为纯蓝图像
        cv2.imwrite(sample1, sample_img_1)  # 保存此图像
        print("默认样本1已补充")
    sample2 = os.path.join(PIC_PATH , "2000000000.png")  # 样本2文件路径
    if not os.path.exists(sample2):
        sample_img_2 = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), np.uint8)  # 创建一个空内容图像
        sample_img_2[:, :, 1] = 255  # 改为纯蓝图像
        cv2.imwrite(sample2, sample_img_2)  # 保存此图像
        print("默认样本2已补充")
    if not os.path.exists(DATA_FILE):
        open(DATA_FILE, "a+")  # 附加读写方式打开文件，达到创建空文件目的
        print("员工信息文件丢失，已重新创建：" + DATA_FILE)
    if not os.path.exists(RECORD_FILE):
        open(RECORD_FILE, "a+")  # 附加读写方式打开文件，达到创建空文件目的
        print("打卡记录文件丢失，已重新创建：" + RECORD_FILE)
    if not os.path.exists(USER_PASSWORD):
        file = open(USER_PASSWORD, "a+", encoding="utf-8")  # 附加读写方式打开文件，达到创建空文件目的
        user = dict()
        user["mr"] = "mrsoft"
        file.write(str(user))  # 将默认管理员账号密码写入到文件中
        file.close()  # 关闭文件
        print("管理员账号密码文件丢失，已重新创建：" + RECORD_FILE)
    if not os.path.exists(WORK_TIME):
        file = open(WORK_TIME, "a+", encoding="utf-8")  # 附加读写方式打开文件，达到创建空文件目的
        file.write("09:00:00/17:00:00")  # 将默认时间写入到文件中
        file.close()  # 关闭文件
        print("上下班时间配置文件丢失，已重新创建：" + RECORD_FILE)


# 加载全部员工信息
def load_employee_info():
    max_id = 1  # 最大员工ID
    try:
        # 使用with语句自动管理文件关闭，避免资源泄漏
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            # 遍历每一行并记录行号，方便定位错误
            for line_num, line in enumerate(file, 1):
                line = line.rstrip()  # 去除换行符

                # 跳过空行
                if not line:
                    print(f"警告：第{line_num}行为空行，已跳过")
                    continue

                # 处理全角逗号（中文逗号）问题
                line = line.replace("，", ",")  # 统一转为半角逗号

                # 分割字段并检查数量
                parts = line.split(",")
                if len(parts) != 3:
                    print(f"错误：第{line_num}行格式错误，预期3个字段，实际{len(parts)}个 → 内容：{line}")
                    continue

                # 解包字段
                id, name, code = parts

                # 检查字段是否为空
                if not id or not name or not code:
                    print(f"错误：第{line_num}行存在空字段 → 内容：{line}")
                    continue

                # 检查ID是否为数字
                if not id.isdigit():
                    print(f"错误：第{line_num}行ID不是数字 → ID：{id}")
                    continue

                # 正常添加员工信息
                o.add(o.Employee(id, name, code))

                # 更新最大ID
                current_id = int(id)
                if current_id > max_id:
                    max_id = current_id

        o.MAX_ID = max_id  # 记录最大ID
        print(f"员工信息加载完成，最大ID：{max_id}")

    except FileNotFoundError:
        print(f"错误：员工信息文件不存在 → {DATA_FILE}")
    except Exception as e:
        print(f"加载员工信息时发生错误：{str(e)}")


# 加载员工图像
def load_employee_pic():
    photos = []  # 样本图像列表
    lables = []  # 标签列表
    try:
        if not os.path.exists(PIC_PATH):
            print(f"照片文件夹不存在：{PIC_PATH}，已跳过照片加载")
            return

        pics = os.listdir(PIC_PATH)  # 读取所有文件
        for file_name in pics:
            # 1. 过滤 macOS 系统隐藏文件 .DS_Store
            if file_name.startswith('.'):  # 所有以点开头的文件都是隐藏文件
                print(f"跳过系统隐藏文件：{file_name}")
                continue

            # 2. 确保处理的是文件（不是子目录）
            file_path = os.path.join(PIC_PATH, file_name)
            if not os.path.isfile(file_path):
                print(f"跳过非文件项：{file_name}")
                continue

            # 3. 过滤非图片文件（只保留.png等图像格式）
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"跳过非图片文件：{file_name}")
                continue

            # 4. 提取并清理特征码
            if len(file_name) < o.CODE_LEN:
                print(f"文件名过短，无法提取特征码：{file_name}，已跳过")
                continue

            code = file_name[:o.CODE_LEN]
            cleaned_code = ''.join(filter(str.isdigit, code))
            if not cleaned_code:
                print(f"特征码无效，已跳过文件：{file_name}（提取的特征码：{code}）")
                continue

            # 5. 读取图像并检查有效性
            img = cv2.imread(file_path, 0)  # 以灰度图读取
            if img is None:
                print(f"图像读取失败，可能文件损坏：{file_name}")
                continue

            # 6. 添加到训练集
            photos.append(img)
            lables.append(int(cleaned_code))

        # 7. 训练识别器（如果有有效照片）
        if photos:
            rs.train(photos, lables)
            print(f"已加载 {len(photos)} 张有效员工照片并完成训练")
        else:
            print("未找到有效员工照片，识别器未训练")

    except Exception as e:
        print(f"加载员工照片时发生错误：{str(e)}")


# 将员工信息持久化
def save_employee_all():
    file = open(DATA_FILE, "w", encoding="utf-8")  # 打开员工信息文件，只写，覆盖
    info = ""  # 待写入的字符串
    for emp in o.EMPLOYEES:  # 遍历所有员工信息
        # 拼接员工信息
        info += str(emp.id) + "," + str(emp.name) + "," + str(emp.code) + "\n"
    file.write(info)  # 将这些员工信息写入到文件中
    file.close()  # 关闭文件


# 删除指定员工的所有照片
def remove_pics(id):
    pics = os.listdir(PIC_PATH)  # 读取所有照片文件
    code = str(hr.get_code_with_id(id))  # 获取该员工的特征码
    for file_name in pics:  # 遍历文件
        if file_name.startswith(code):  # 如果文件名以特征码开头
            os.remove(PIC_PATH + file_name)  # 删除此文件
            print("删除照片：" + file_name)


# 加载所有打卡记录
def load_lock_record():
    file = open(RECORD_FILE, "r", encoding="utf-8")  # 打开打卡记录文件，只读
    text = file.read()  # 读取所有文本
    if len(text) > 0:  # 如果存在文本
        o.LOCK_RECORD = eval(text)  # 将文本转换成打卡记录字典
    file.close()  # 关闭文件


# 将打卡记录持久化
def save_lock_record():
    file = open(RECORD_FILE, "w", encoding="utf-8")  # 打开打卡记录文件，只写，覆盖
    info = str(o.LOCK_RECORD)  # 将打卡记录字典转换成字符串
    file.write(info)  # 将字符串内容写入到文件中
    file.close()  # 关闭文件


# 将上下班时间写到文件中
def save_work_time_config():
    file = open(WORK_TIME, "w", encoding="utf-8")  # 打开打卡记录文件，只写，覆盖
    times = str(o.WORK_TIME) + "/" + str(o.CLOSING_TIME)
    file.write(times)  # 将字符串内容写入到文件中
    file.close()  # 关闭文件


# 加载上下班时间数据
def load_work_time_config():
    file = open(WORK_TIME, "r", encoding="utf-8")  # 打开打卡记录文件，只读
    text = file.read().rstrip()  # 读取所有文本
    times = text.split("/")  # 分割字符串
    o.WORK_TIME = times[0]  # 第一个值是上班时间
    o.CLOSING_TIME = times[1]  # 第二个值是下班时间
    file.close()  # 关闭文件


# 加载管理员账号密码
def load_users():
    file = open(USER_PASSWORD, "r", encoding="utf-8")  # 打开打卡记录文件，只读
    text = file.read()  # 读取所有文本
    if len(text) > 0:  # 如果存在文本
        o.USERS = eval(text)  # 将文本转换成打卡记录字典
    file.close()  # 关闭文件


# 生成csv文件，采用Windows默认的gbk编码
def create_CSV(file_name, text):
    file = open(PATH + file_name + ".csv", "w", encoding="gbk")  # 打开文件，只写，覆盖
    file.write(text)  # 将文本写入文件中
    file.close()  # 关闭文件
    print("已生成文件，请注意查看：" + PATH + file_name + ".csv")
