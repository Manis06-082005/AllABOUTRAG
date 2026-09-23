class ContextBuilder:
    def build(self,results):
        context='\n\n'.join(
            result['text'] for result in results
        )
        return context