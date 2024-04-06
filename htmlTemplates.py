css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background: linear-gradient(#9896F0, #FBC8D5);
}
.chat-message.bot {
    background: linear-gradient(#71C5EE, #025091);
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}

'''

bot_template = '''
<div class="chat-message bot" style="color: black;">
    <div class="avatar">
        <img src="https://img.freepik.com/premium-vector/chat-bot-vector-logo-design-concept_418020-241.jpg" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user" style="color: black;">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/256/4825/4825038.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

sidebar_custom_css = """
<style>
/* Set background color of sidebar */
[data-testid="stSidebar"] {
    background-color: #FEBE9A; /* Change this color to your desired background color */
}
</style>
"""



