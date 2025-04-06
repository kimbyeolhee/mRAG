"""
Prompt 클래스 정의
생성 모델에서 사용할 프롬프트 템플릿을 관리합니다.
"""

class Prompt:
    """
    생성 모델의 프롬프트 템플릿을 담는 클래스
    """
    def __init__(self, 
                 system=None, 
                 user=None, 
                 system_without_docs=None, 
                 user_without_docs=None):
        """
        Prompt 클래스 초기화
        
        Args:
            system: 문서가 있을 때 사용하는 시스템 메시지
            user: 문서가 있을 때 사용하는 사용자 메시지
            system_without_docs: 문서가 없을 때 사용하는 시스템 메시지
            user_without_docs: 문서가 없을 때 사용하는 사용자 메시지
        """
        # 기본값 설정
        self.system = system or "You are a helpful AI assistant."
        self.user = user or "\"question = '{question}'\ndocuments = '''{docs}'''\nAnswer the question based on the documents.\""
        self.system_without_docs = system_without_docs or "You are a helpful AI assistant."
        self.user_without_docs = user_without_docs or "\"question = '{question}'\nAnswer the question based on your knowledge.\""

    @classmethod
    def from_dict(cls, data):
        """
        딕셔너리에서 Prompt 객체 생성
        
        Args:
            data: 프롬프트 정보가 담긴 딕셔너리
            
        Returns:
            Prompt 객체
        """
        return cls(
            system=data.get('system'),
            user=data.get('user'),
            system_without_docs=data.get('system_without_docs'),
            user_without_docs=data.get('user_without_docs')
        )

    def to_dict(self):
        """
        Prompt 객체를 딕셔너리로 변환
        
        Returns:
            딕셔너리 형태의 프롬프트 정보
        """
        return {
            'system': self.system,
            'user': self.user,
            'system_without_docs': self.system_without_docs,
            'user_without_docs': self.user_without_docs
        } 