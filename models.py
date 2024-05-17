from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey



Base = declarative_base()

class Unofarm(Base):
    __tablename__ = "unofarm"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    content = Column(Text, nullable=False)
    description = Column(Text)
    file_path = Column(String(255), nullable=False)
    url = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
 



    videos = relationship("Video", backref="unofarm")

class Video(Base):
    __tablename__ = "video"

    VNO = Column(Integer, primary_key=True)
    V_TITLE = Column(String(200))
    PATHUPLOAD = Column(String(500))
    FILETYPE = Column(String(50))
    REGDATE = Column(DateTime, default=datetime.now)
    DNO = Column(Integer, ForeignKey("unofarm.id"))  # Unofarm 테이블의 id 컬럼과 연결

    # Unofarm 객체로의 역참조 설정
    unofarm = relationship("Unofarm", backref="video")