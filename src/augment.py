"""
Data Augmentation for Hinglish E-Commerce Complaints (v2)
Generates synthetic Hinglish complaints with urgency-appropriate templates.
"""
import random
import pandas as pd
import os

random.seed(42)

# ============================================================================
# URGENCY-SPECIFIC TEMPLATES
# ============================================================================

# HIGH URGENCY: threats, escalation, exclamation, "urgent", strong language
HIGH_URGENCY_TEMPLATES = {
    'Order_Status': [
        "Mera order {time_word} se stuck hai! URGENT hai, {urgency_word}!",
        "Order {status_word} hai, consumer court me jaunga agar {time_word} tak resolve nahi hua!",
        "MANAGER SE BAAT KARO! Order {num} din se {status_word} hai!",
        "Order {status_word} hai, {urgency_word}! Legal action lunga!",
        "Order {status_word} hai but koi response nahi! {urgency_word}!",
        "Order {status_word} hai, {time_word} se pareshan hu! TURANT KARO!",
        "Order tracking {status_word} hai, IMMEDIATELY fix karo warna consumer forum me complaint karunga!",
        "Order {num} din se {status_word} hai! Bahut urgent hai! {urgency_word}!",
        "Order confirm nahi ho raha, {time_word} se try kar raha hu! ESCALATE KARO!",
        "Order {status_word} hai, paisa kat gaya but kuch nahi mila! FRAUD hai! {urgency_word}!",
        "Order {status_word} hai but delivery date cross ho gayi! {urgency_word}!",
        "Order status {status_word} hai, customer care bhi help nahi kar raha! MANAGER KO BULAO!",
        "Order {num} baar {status_word} ho raha hai! Fix karo ya paisa wapas karo!",
        "Order {status_word} hai, {urgency_word}! Nahi toh police complaint karunga!",
        "Order {time_word} pe place kiya tha, abhi tak {status_word}! BAHUT URGENT!",
        "Order cancel karna hai but option {status_word} hai! {urgency_word}!",
        "Order dispatch ho gaya but address galat hai! IMMEDIATELY CHANGE KARO!",
        "Order {status_word} hai, main bahut gussa hu! {urgency_word}!",
        "Order {num} din se transit me hai! Kya ho raha hai? URGENT!",
        "Order status {status_word} hai, {urgency_word}! Mera paisa wapas karo!",
    ],
    'Delivery_Issue': [
        "Delivery boy ne package fek ke diya! {urgency_word}!",
        "Delivery {time_word} se delay hai! URGENT resolve karo!",
        "Delivery wrong address pe ho gayi! {urgency_word}!",
        "Delivery boy ne extra money manga! FRAUD hai! {urgency_word}!",
        "Package delivered but {damage_word} hai! {urgency_word}!",
        "Delivery {time_word} se nahi ho rahi! Consumer court me jaunga!",
        "Delivery boy rude tha! MANAGER SE BAAT KARO! {urgency_word}!",
        "Package {damage_word} hai! IMMEDIATELY refund karo!",
        "Delivery instructions follow nahi kiye! {urgency_word}!",
        "Delivery agent ne wrong person ko de diya! FRAUD! {urgency_word}!",
        "Package missing hai! {urgency_word}! Police complaint karunga!",
        "Delivery {num} din se delay hai! BAHUT URGENT!",
        "Delivery boy ne OTP maanga without delivery! SCAM hai! {urgency_word}!",
        "Package {damage_word} hai, {urgency_word}! TURANT REPLACE KARO!",
        "Delivery date {time_word} thi but abhi tak nahi aayi! ESCALATE KARO!",
        "Delivery boy ne security guard ko de diya! {urgency_word}!",
        "Package delivered but seal broken hai! {urgency_word}!",
        "Delivery {time_word} se pending hai! Koi action nahi le raha!",
        "Delivery boy demanded cash on delivery! Already paid tha! FRAUD!",
        "Package {damage_word} hai! {urgency_word}! Full refund do!",
    ],
    'Returns_Refunds': [
        "Refund {time_word} se nahi mila! {urgency_word}!",
        "Return request {time_word} se pending hai! APPROVE KARO!",
        "Refund amount galat aaya! {urgency_word}! Correct karo!",
        "Return reject kar diya! Consumer court me jaunga! {urgency_word}!",
        "Refund {time_word} pe initiate hua but abhi tak nahi aaya! URGENT!",
        "Return pickup {time_word} se nahi ho raha! {urgency_word}!",
        "Refund completed bol rahe ho but paisa nahi aaya! FRAUD hai!",
        "Return item {damage_word} tha! Full refund do! {urgency_word}!",
        "Refund {num} din se pending hai! BAHUT URGENT!",
        "Return policy ke according refund milna chahiye! {urgency_word}!",
        "Refund ₹{amount} expected tha but ₹{amount2} aaya! CORRECT KARO!",
        "Return request {time_word} pe ki thi! {urgency_word}! Koi response nahi!",
        "Refund status check kaise karu? Koi option nahi! {urgency_word}!",
        "Return item wapas bhej diya! Refund kab aayega? URGENT!",
        "Refund amount bank mein credit nahi ho raha! {urgency_word}!",
        "Return request approve karo! Item {damage_word} hai! TURANT!",
        "Refund {time_word} se process ho raha hai! KAB HOGA? URGENT!",
        "Return item ka refund nahi mil raha! {urgency_word}!",
        "Refund ₹{amount} hona chahiye! {urgency_word}! Check karo!",
        "Return {time_word} pe deliver hua! Refund status batao! URGENT!",
    ],
    'Payment_Invoice': [
        "Payment double charge hua hai! {urgency_word}! Refund karo!",
        "Payment fail ho gaya but paisa kat gaya! {urgency_word}!",
        "Invoice galat hai! {urgency_word}! Correct karo!",
        "Payment ₹{amount} kat gaya but order nahi hua! FRAUD hai!",
        "Payment {status_word} hai but order {status_word2} hai! {urgency_word}!",
        "Invoice me GST galat hai! {urgency_word}! Legal issue hai!",
        "Payment gateway pe amount stuck hai! {urgency_word}!",
        "Payment successful but merchant ko nahi pahuncha! {urgency_word}!",
        "Invoice {time_word} se generate nahi ho raha! URGENT!",
        "Payment {amount} kiya but {amount2} kat gaya! {urgency_word}!",
        "Payment fail ho gaya but wallet balance cut gaya! {urgency_word}!",
        "Payment {time_word} se pending hai! {urgency_word}!",
        "Invoice me extra charges lagaye hai! {urgency_word}! Remove karo!",
        "Payment successful but order cancel dikha raha hai! {urgency_word}!",
        "Payment double hua hai! {urgency_word}! Refund karo!",
        "Invoice download nahi ho raha! {urgency_word}!",
        "Payment gateway frozen hai! {urgency_word}! Release karo!",
        "Payment {status_word} hai but confirmation nahi mila! URGENT!",
        "Invoice amount galat hai! {urgency_word}! Recalculate karo!",
        "Payment {time_word} pe kiya but order confirm nahi hua! {urgency_word}!",
    ],
    'Account_Technical': [
        "Account {status_word} hai! {urgency_word}!",
        "App {status_word} ho raha hai! {urgency_word}! Fix karo!",
        "Login nahi ho raha! {urgency_word}! {num} baar try kiya!",
        "OTP nahi aa raha! {urgency_word}! Multiple times try kiya!",
        "Account {status_word} hai but maine kuch nahi kiya! {urgency_word}!",
        "App crash ho raha hai! {urgency_word}! TURANT fix karo!",
        "Account security issue hai! Someone else login kar raha! {urgency_word}!",
        "App {time_word} se {status_word} hai! BAHUT URGENT!",
        "Account {status_word} hai, customer care bhi help nahi kar raha! ESCALATE KARO!",
        "Login {status_word} hai! Password sahi hai! {urgency_word}!",
        "Account {status_word} hai! Koi explanation nahi diya! {urgency_word}!",
        "App {status_word} hai! Data load nahi ho raha! URGENT!",
        "Account {time_word} se {status_word} hai! Help karo!",
        "App update ke baad sab {status_word} ho gaya! {urgency_word}!",
        "Account {status_word} hai! New device pe login nahi ho raha! URGENT!",
        "OTP {status_word} hai but verify nahi ho raha! {urgency_word}!",
        "Account delete karna hai but option nahi mil raha! {urgency_word}!",
        "App {status_word} hai, payment nahi ho raha! URGENT!",
        "Account {status_word} hai but koi reason nahi bataya! {urgency_word}!",
        "App {time_word} se slow hai! BAHUT URGENT hai!",
    ],
    'Wrong_Damaged_Product': [
        "Wrong product aaya hai! {urgency_word}! Exchange karo!",
        "Product {damage_word} hai! {urgency_word}! Refund do!",
        "Product {damage_word} hai! Consumer court me jaunga! {urgency_word}!",
        "Wrong item deliver hua! {urgency_word}! Turant replace karo!",
        "Product {damage_word} hai! {urgency_word}! Full refund do!",
        "Product {damage_word} hai! {urgency_word}! Legal action lunga!",
        "Wrong product aaya hai! {urgency_word}! Paisa wapas karo!",
        "Product missing hai! Box {damage_word} tha! {urgency_word}!",
        "Product {damage_word} hai! {urgency_word}! TURANT ACTION LO!",
        "Wrong product aaya hai! {time_word} se pehle replace karo!",
        "Product {damage_word} hai! Fire hazard hai! {urgency_word}!",
        "Wrong item aaya hai! {urgency_word}! ESCALATE KARO!",
        "Product {damage_word} hai! {urgency_word}! Koi response nahi!",
        "Product {damage_word} hai! Completely unusable hai! URGENT!",
        "Wrong product deliver hua! {urgency_word}! Correct item bhejo!",
        "Product {damage_word} hai! {urgency_word}! Immediate action lo!",
        "Product duplicate hai! Fake hai! {urgency_word}!",
        "Product {damage_word} hai! {urgency_word}! Replace ya refund do!",
        "Wrong product aaya hai! {urgency_word}! Bahut gussa hu!",
        "Product {damage_word} hai! {urgency_word}! Consumer forum me complaint karunga!",
        "Mera order {damage_word} aaya hai! {urgency_word}! Replacement chahiye!",
        "Order {damage_word} product aaya hai! {urgency_word}! Exchange karo!",
        "Mera order wrong item aaya hai! {urgency_word}! Sahi product bhejo!",
        "Order delivered but product {damage_word} hai! {urgency_word}!",
        "Mera order ka product {damage_word} hai! Refund ya replacement do!",
        "Order me wrong product aaya hai! {urgency_word}! Check karo!",
        "Mera order {damage_word} ho ke aaya hai! {urgency_word}!",
        "Order ka product {damage_word} hai! {urgency_word}! Turant replace karo!",
        "Mera order me missing items hain! {urgency_word}! Complete karo!",
        "Order {damage_word} hai! mujhe replacement chahiye! {urgency_word}!",
        "Mera order me wrong item aaya hai! Exchange karo jaldi!",
        "Order ka product kharab hai! {urgency_word}! Refund do!",
        "Mera order delivered but product {damage_word} hai!",
        "Order wrong product aaya hai! {urgency_word}! Sahi wala bhejo!",
        "Mera order ka item {damage_word} hai! {urgency_word}!",
        "Order me se kuch items missing hain! {urgency_word}!",
        "Mera order {damage_word} ho ke aaya! Replacement chahiye!",
        "Order product {damage_word} hai! {urgency_word}! Action lo!",
        "Mera order ka product nahi chal raha! {urgency_word}!",
        "Order delivered wrong item! {urgency_word}! Exchange karo!",
        "Mera order {damage_word} product aaya! {urgency_word}!",
    ],
    'Customer_Service': [
        "Customer care bilkul useless hai! {urgency_word}! MANAGER SE BAAT KARO!",
        "{num} baar call kiya koi uthata nahi! {urgency_word}!",
        "Customer care ne rude behaviour kiya! ESCALATE KARO! {urgency_word}!",
        "Support agent ne fake promise kiya! {urgency_word}! ACTION LO!",
        "Customer care number kaam nahi kar raha! {urgency_word}!",
        "Email kiya tha {time_word} se koi response nahi! URGENT!",
        "Customer care se koi resolution nahi mil raha! {urgency_word}!",
        "Chatbot useless hai! Human se baat karo! {urgency_word}!",
        "Customer care ne transfer kiya {num} baar! {urgency_word}!",
        "Social media pe post karunga agar reply nahi aaya! {urgency_word}!",
        "Customer care promised callback but kabhi nahi aaya! FRAUD! {urgency_word}!",
        "Customer support se baat nahi ho raha! {urgency_word}! ESCALATE KARO!",
        "Customer care agent ne hang up kar diya! {urgency_word}!",
        "Twitter pe complaint karunga agar {time_word} tak resolve nahi hua! {urgency_word}!",
        "Customer care bilkul bekar hai! Paisa barbad! {urgency_word}!",
        "Customer care ne galat information di! {urgency_word}!",
        "Customer care se baat karke bhi kuch nahi hua! {urgency_word}!",
        "Customer care ka wait time {num} ghanta hai! {urgency_word}!",
        "Customer care ne case close kar diya bina resolution ke! {urgency_word}!",
        "Customer care bilkul deaf hai! Koi sunta nahi! {urgency_word}!",
    ],
    'Product_Quality': [
        "Product {time_word} mein kharab ho gaya! {urgency_word}! Refund do!",
        "Quality bilkul bakwas hai! {urgency_word}! Paisa barbad!",
        "Product {damage_word} ho gaya after {time_word} use! {urgency_word}!",
        "Material cheap hai! {urgency_word}! Consumer court me jaunga!",
        "Product ka quality listing se bilkul alag hai! FRAUD! {urgency_word}!",
        "Product stopped working! {urgency_word}! Replace karo!",
        "Quality itni ghatiya hai! {urgency_word}! Legal action lunga!",
        "Product {damage_word} after {num} days! {urgency_word}! Refund ya replace!",
        "Battery backup zero hai! {urgency_word}! Cheating hai ye!",
        "Product {damage_word} hai! {urgency_word}! Consumer forum me complaint!",
        "Quality inferior hai! {urgency_word}! Paisa wapas karo!",
        "Product fake lag raha hai! {urgency_word}! Original bhejo!",
        "Material {damage_word} hai! {urgency_word}! Immediate action!",
        "Product {damage_word} after first use! {urgency_word}! Fraud!",
        "Quality listing se different hai! {urgency_word}! Cheating!",
        "Product {damage_word} within {num} days! {urgency_word}! Replace!",
        "Quality cheap hai! {urgency_word}! Paisa barbad kar diya!",
        "Product {damage_word}! {urgency_word}! Consumer court jaunga!",
        "Material quality bilkul zero hai! {urgency_word}! Refund do!",
        "Product {damage_word} ho gaya! {urgency_word}! Fraud hai ye!",
    ],
    'Pricing_Discount': [
        "Price checkout pe badh gaya! {urgency_word}! FRAUD hai!",
        "Coupon apply nahi ho raha! {urgency_word}! Paisa wapas karo!",
        "Hidden charges add ho gaye! {urgency_word}! Cheating!",
        "Discount nahi mila jo dikha tha! {urgency_word}! Legal action!",
        "Price suddenly badh gaya! {urgency_word}! Fraud!",
        "Cashback nahi mila! {urgency_word}! Paisa barbad!",
        "Extra charges kahan se aaye? {urgency_word}! Consumer court!",
        "Coupon code kaam nahi kar raha! {urgency_word}! Fix karo!",
        "Price app pe alag website pe alag! {urgency_word}! Cheating!",
        "Loyalty points credit nahi hue! {urgency_word}! FRAUD!",
        "Promotional price nahi mila! {urgency_word}! Legal action!",
        "Price match guarantee nahi di! {urgency_word}! Consumer court!",
        "Delivery charge last pe add ho gaya! {urgency_word}! Fraud!",
        "Membership benefits apply nahi hue! {urgency_word}! Cheating!",
        "Price {time_word} mein double ho gaya! {urgency_word}! Scam!",
        "EMI pe extra interest lagaya! {urgency_word}! Fraud!",
        "Coupon fraud hai! {urgency_word}! Paisa wapas karo!",
        "Hidden fees dikhai nahi di thi! {urgency_word}! Cheating!",
        "Discount fake hai! {urgency_word}! Consumer court me jaunga!",
        "Price checkout pe change ho gaya! {urgency_word}! Scam hai ye!",
    ],
}

# MEDIUM URGENCY: complaints about issues, firm but not threatening
MEDIUM_URGENCY_TEMPLATES = {
    'Order_Status': [
        "Order {status_word} hai, kya problem hai?",
        "Order {time_word} se {status_word} hai, update do",
        "Order status {status_word} hai, expected date kab hai?",
        "Order {status_word} hai but confirmation email nahi aaya",
        "Order tracking {status_word} hai, koi batao kya ho raha hai",
        "Order {num} din se transit mein hai, kab milega?",
        "Order status update nahi ho raha, please check karo",
        "Order place kiya tha but abhi tak {status_word}",
        "Order {status_word} hai but delivery date cross ho gayi",
        "Order me missing item hai, check karo",
        "Order {num} items ka tha but sirf {num2} aaye",
        "Order delivered dikh raha hai but mujhe kuch nahi mila",
        "Order tracking not working properly, blank page aa raha hai",
        "Order status inconsistent hai, ek jagah delivered aur dusri pe pending",
        "Order placed but wrong phone number add ho gaya",
        "Order confirm hone ke baad delivery date change ho gayi",
        "Order status pending hai but payment already ho chuka hai",
        "Order cancel karne ka option nahi aa raha",
        "Order {time_word} pe place kiya, expected date kab hai?",
        "Order {status_word} hai, mujhe urgent delivery chahiye",
    ],
    'Delivery_Issue': [
        "Delivery {time_word} se delay ho rahi hai, kab aayegi?",
        "Delivery status delivered dikha raha hai but package nahi mila",
        "Delivery date {num} din se delay hai, koi update do",
        "Delivery boy ne package fek ke diya, bahut damage hai",
        "Delivery instructions follow nahi kiye, gate pe chhod ke chala gaya",
        "Delivery {status_word} hai but expected date cross ho gayi",
        "Package delivered but kisi aur ke naam pe dikha raha hai",
        "Delivery ke baad package {damage_word} nikla",
        "Delivery {time_word} ho rahi hai, pehle {time_word2} se hoti thi",
        "Delivery scheduled for {time_word} but nobody came",
        "Package delivered but outer box completely crushed tha",
        "Delivery agent refused to come to my floor",
        "Delivery {num} din lag rahe hain, pehle {num2} din mein aata tha",
        "Delivery boy demanded extra money for delivery",
        "Wrong address pe deliver ho gaya, mera address toh {address_word} hai",
        "Delivery instructions diye the gate pe rakhne ko lekin kisi aur ne le liya",
        "Delivery ke baad product missing hai, empty box aaya",
        "Delivery person ne OTP maanga but maine diya nahi",
        "Package delivered but seal already broken tha",
        "Delivery date change ho gayi bina meri marzi ke",
    ],
    'Returns_Refunds': [
        "Refund kab milega? {time_word} se wait kar raha hu",
        "Return request {time_word} se pending hai, approve karo",
        "Refund {status_word} hai but paisa abhi tak nahi aaya",
        "Return reject kar diya bina reason ke",
        "Refund amount galat aaya hai, ₹{amount} instead of ₹{amount2}",
        "Refund kab tak milega? Bank account pe transfer karo",
        "Return pickup {time_word} se nahi ho raha, schedule karo",
        "Refund completed dikh raha hai but account mein credit nahi hua",
        "Return item {time_word} se return karne ki koshish kar raha hu",
        "Refund ₹{amount} ka hona chahiye but sirf ₹{amount2} aaya",
        "Return approved but refund kab aayega? {time_word} ho gaya",
        "Return item damaged aaya tha, refund chahiye",
        "Refund {time_word} pe initiate hua tha but abhi tak nahi aaya",
        "Return request {time_word} pe ki thi but ab tak koi update nahi",
        "Refund amount ₹{amount} hona chahiye, check karo",
        "Return item deliver ho chuka hai, refund process start karo",
        "Return ke baad product wapas nahi bhej pa raha, help karo",
        "Refund ke liye kitna wait karna padega? {time_word} ho gaya",
        "Return request reject without inspection, please reopen",
        "Refund amount ₹{amount} tha but sirf ₹{amount2} credit hua",
    ],
    'Payment_Invoice': [
        "Payment {status_word} ho gaya but order {status_word2} hai",
        "Payment fail ho gaya but paisa kat gaya, please check",
        "Invoice {status_word} hai, correct invoice bhejo",
        "Payment successful dikha raha hai but order {status_word} hai",
        "Invoice me GST number galat hai, update karo",
        "Payment double charge hua hai, refund karo",
        "Invoice download nahi ho raha, {status_word} hai page",
        "Payment {amount} ka hua tha but {amount2} kat gaya",
        "Invoice me address galat hai, update karo",
        "Payment gateway pe amount stuck hai, release karo",
        "Invoice {time_word} se generate nahi ho raha",
        "Payment {status_word} but confirmation nahi mila",
        "Invoice me price aur paid amount match nahi kar rahe",
        "Payment {status_word} hai but order status mein koi update nahi",
        "Payment ke baad order cancel ho gaya, refund do",
        "Invoice me service charge extra lagaya hai, remove karo",
        "Payment successful but merchant ko payment nahi pahuncha",
        "Payment {time_word} pe ki thi but abhi tak order confirm nahi hua",
        "Payment {amount} kiya but {amount2} charge ho gaya",
        "Invoice me customer name galat hai, fix karo",
    ],
    'Account_Technical': [
        "Account {status_word} ho gaya hai, please help",
        "App {status_word} ho raha hai, {time_word} se use nahi ho raha",
        "Login nahi ho raha, {status_word} dikha raha hai",
        "OTP nahi aa raha, {num} baar try kiya",
        "Account {status_word} hai, koi explanation nahi diya",
        "App pe error aa raha hai, {status_word} hai page",
        "Password reset link {status_word} hai, naya password set nahi ho raha",
        "Account verify nahi ho raha, {time_word} se pending hai",
        "App {status_word} ho gaya hai after update, fix karo",
        "Login credentials {status_word} dikha raha hai",
        "App crash ho raha hai, {time_word} se bar bar ho raha hai",
        "Profile update nahi ho raha, {status_word} hai form",
        "Account security issue hai, someone else is logging in",
        "App {status_word} hai, data load nahi ho raha",
        "OTP {status_word} hai but verify nahi ho raha",
        "App {status_word} hai, payment process nahi ho raha",
        "Account {status_word} hai but koi response nahi mil raha",
        "App update ke baad sab {status_word} ho gaya hai",
        "Account {status_word} hai, KYC verify nahi ho raha",
        "App {status_word} hai, coupon apply nahi ho raha",
    ],
    'Wrong_Damaged_Product': [
        "Wrong product aaya hai, exchange karo",
        "Product {damage_word} hai, refund do",
        "Wrong item deliver hua hai, mene {product_word} order kiya tha",
        "Product {damage_word} tha, delivery ke baad pata chala",
        "Product {damage_word} hai, replacement chahiye",
        "Wrong product bheja hai, {product_word} ki jagah {product_word2} aaya",
        "Product missing hai, box {damage_word} tha but andar kuch nahi",
        "Wrong color aaya hai, maine {color_word} order kiya tha",
        "Product {damage_word} hai, pura {damage_word} ho gaya hai",
        "Product duplicate lag raha hai, original nahi hai",
        "Wrong size aaya hai, {size_word} manga tha but {size_word2} aaya",
        "Product {damage_word} hai, {num} din pehle receive hua",
        "Product {damage_word} hai, ek piece missing hai",
        "Product {damage_word} hai, battery {damage_word} ho gayi hai",
        "Product incomplete hai, {num} parts missing hain",
        "Product {damage_word} hai, completely unusable hai",
        "Wrong item deliver hua, fashion item tha but kitchen item aaya",
        "Product {damage_word} hai, {urgency_word}",
        "Product not as described hai, please check",
        "Product {damage_word} hai, {time_word} ke andar replace karo",
        "Mera order {damage_word} aaya hai, replacement chahiye",
        "Order {damage_word} product aaya hai, exchange karo",
        "Mera order wrong item aaya hai, sahi product bhejo",
        "Order delivered but product {damage_word} hai, check karo",
        "Mera order ka product {damage_word} hai, refund ya replacement do",
        "Order me wrong product aaya hai, please check karo",
        "Mera order {damage_word} ho ke aaya hai, replace karo",
        "Order ka product {damage_word} hai, turant replace karo",
        "Mera order me missing items hain, complete karo",
        "Order {damage_word} hai, mujhe replacement chahiye",
        "Mera order me wrong item aaya hai, exchange karo jaldi",
        "Order ka product kharab hai, refund do please",
        "Mera order delivered but product {damage_word} hai",
        "Order wrong product aaya hai, sahi wala bhejo",
        "Mera order ka item {damage_word} hai, check karo",
        "Order me se kuch items missing hain, please complete karo",
        "Mera order {damage_word} ho ke aaya, replacement chahiye abhi",
        "Order product {damage_word} hai, action lo please",
        "Mera order ka product nahi chal raha, help karo",
        "Order delivered wrong item, exchange karo please",
        "Mera order {damage_word} product aaya hai, theek karo",
    ],
    'Customer_Service': [
        "Customer care se response nahi mil raha, please help",
        "Customer care pe call kiya but koi resolution nahi mila",
        "Support agent rude tha, please take action",
        "Customer care promised callback but kabhi nahi aaya",
        "Customer care number busy rehta hai, please help",
        "Customer care ne proper answer nahi diya",
        "Email kiya but 3 din se koi reply nahi",
        "Customer care se baat nahi ho pa rahi",
        "Customer care ka response time bahut zyada hai",
        "Support team se koi help nahi mili",
        "Customer care ne case forward kiya but kuch nahi hua",
        "Customer care ka response delayed hai",
        "Customer care agent ne galat info di",
        "Customer care se baat karke bhi problem solve nahi hui",
        "Customer care ka wait time bahut zyada hai",
        "Customer care ne query ignore kar di",
        "Customer care se proper guidance nahi mili",
        "Customer care ka response unsatisfactory hai",
        "Customer care ne issue escalate nahi kiya",
        "Customer care ka behavior acha nahi tha",
    ],
    'Product_Quality': [
        "Product ka quality acha nahi hai, please check",
        "Product {time_word} mein kharab ho gaya",
        "Material quality listing se kam hai",
        "Product ka color fade ho raha hai after wash",
        "Product {damage_word} ho gaya after {time_word} use",
        "Quality expected se kam hai, please help",
        "Product ki material cheap lag rahi hai",
        "Product ka stitching nikal raha hai",
        "Battery backup kam hai expected se",
        "Product ki performance degrade ho gayi hai",
        "Product ka size listing se alag hai",
        "Material quality inferior hai expected se",
        "Product ki finish kharab hai",
        "Product {damage_word} after regular use",
        "Quality expected se different hai",
        "Product ka weight kam hai mentioned se",
        "Product ki color alag hai listing se",
        "Material quality cheap feel ho raha hai",
        "Product durability kam hai expected se",
        "Product ka performance slow hai",
    ],
    'Pricing_Discount': [
        "Price checkout pe thoda zyada aaya, please check",
        "Coupon apply ho raha hai but discount nahi mil raha",
        "Delivery charges kahan se add ho gaye?",
        "Discount percentage listing se kam hai",
        "Price same hai sale ke baad bhi",
        "Cashback kab tak milega?",
        "Extra charges ka breakdown batao",
        "Coupon code valid hai but apply nahi ho raha",
        "Price app pe aur website pe alag hai",
        "Loyalty points kab credit honge?",
        "Promotional price kab tak milega?",
        "Price match guarantee kaise claim karu?",
        "Delivery charge waiving ka kya rule hai?",
        "Membership discount kab apply hoga?",
        "Price increase kyun hua suddenly?",
        "EMI options kya kya hai?",
        "Coupon code ka expiry kya hai?",
        "Hidden charges ka detail batao",
        "Discount applicable hai ya nahi?",
        "Price checkout pe different kyun hai?",
    ],
}

# LOW URGENCY: questions, general inquiries, polite, no urgency
LOW_URGENCY_TEMPLATES = {
    'Order_Status': [
        "Refund usually kitne working days leta h?",
        "Order status kaise check karu?",
        "Order cancel karne ka option kahan hai?",
        "Expected delivery date kya hai?",
        "Order place kaise karu?",
        "Order me address change kar sakte hai kya?",
        "Order ka tracking number kahan milega?",
        "Order confirm hone ke baad cancel kar sakte hai kya?",
        "Order {num} items ka hai, sab ek saath aayenge kya?",
        "Order status me kya dikha raha hai?",
        "Order place karne ke baad payment change kar sakte hai kya?",
        "Order me kitne items hai, confirm karo",
        "Order dispatch hone ke baad address change ho sakta hai kya?",
        "Order status check karne ka tarika batao",
        "Order ka expected delivery date kya hai?",
        "Order confirm ho gaya hai, ab kya hoga?",
        "Order me kya kya aayega, list bhejo",
        "Order place karne ke baad phone number change ho sakta hai kya?",
        "Order status update kaise hota hai?",
        "Order ka bill kaise download karu?",
    ],
    'Delivery_Issue': [
        "Delivery usually kitne din leti h?",
        "Delivery address change kar sakte hai kya?",
        "Delivery date extend ho sakti hai kya?",
        "Delivery boy ko instruction kaise de?",
        "Delivery ke liye kya kya documents chahiye?",
        "Delivery slot book kaise karu?",
        "Delivery ke baad product return ho sakta hai kya?",
        "Delivery time change ho sakta hai kya?",
        "Delivery ke liye koi extra charge hai kya?",
        "Delivery instructions kahan de sakta hu?",
        "Delivery boy ko tip de sakte hai kya?",
        "Delivery ke baad product check kaise karu?",
        "Delivery schedule kaise dekhu?",
        "Delivery ke liye kya kya items allowed hai?",
        "Delivery boy ko message kaise karu?",
        "Delivery ke baad product exchange ho sakta hai kya?",
        "Delivery ke liye kya kya chahiye?",
        "Delivery instructions diye the but follow nahi hue",
        "Delivery ke baad product ka status kaise check karu?",
        "Delivery ke liye kya kya kar sakta hu?",
    ],
    'Returns_Refunds': [
        "Refund usually kitne working days leta h?",
        "Return policy kya hai?",
        "Return item kaise bheju?",
        "Refund kaise check karu?",
        "Return request kaise karu?",
        "Refund kitne din me aata hai?",
        "Return item ka packaging kaise karu?",
        "Refund kahan aata hai, bank ya wallet?",
        "Return item ko kahan bheju?",
        "Refund status kaise check karu?",
        "Return request kab tak kar sakte hai?",
        "Refund ke liye kya kya chahiye?",
        "Return item ka label kaise print karu?",
        "Refund process kaise hota hai?",
        "Return item ko kaise pack karu?",
        "Refund ke liye koi charge hai kya?",
        "Return item ko kahan drop karu?",
        "Refund status update kaise hota hai?",
        "Return request approve hone me kitna time lagta hai?",
        "Refund ke liye kya kya documents chahiye?",
    ],
    'Payment_Invoice': [
        "Payment usually kitne din me confirm hota hai?",
        "Invoice kaise download karu?",
        "Payment methods kya kya hai?",
        "Invoice me kya kya details hoti hai?",
        "Payment fail ho gaya toh kya karu?",
        "Invoice change ho sakta hai kya?",
        "Payment ke baad order confirm hota hai kya?",
        "Invoice ko kaise save karu?",
        "Payment process kaise hota hai?",
        "Invoice ke liye kya kya chahiye?",
        "Payment ke baad kya kya hota hai?",
        "Invoice ko kaise share karu?",
        "Payment ke liye kya kya options hai?",
        "Invoice me GST kaise dekhu?",
        "Payment ke baad order status update hota hai kya?",
        "Invoice ko kaise print karu?",
        "Payment ke liye kya kya steps hai?",
        "Invoice ke liye koi charge hai kya?",
        "Payment ke baad kya kya hota hai?",
        "Invoice ko kaise verify karu?",
    ],
    'Account_Technical': [
        "Account kaise banau?",
        "Password kaise reset karu?",
        "Profile kaise update karu?",
        "Account delete kaise karu?",
        "Login kaise karu?",
        "OTP kaise resend karu?",
        "Account verify kaise karu?",
        "Profile name kaise change karu?",
        "Account settings kahan hai?",
        "Password kaise change karu?",
        "Account me kya kya update kar sakte hai?",
        "Login credentials kya hai?",
        "Account security kaise check karu?",
        "Profile photo kaise lagau?",
        "Account me kya kya details hai?",
        "Password kaise strong banau?",
        "Account ko kaise secure karu?",
        "Profile kaise dekhu?",
        "Account me kya kya options hai?",
        "Account kaise manage karu?",
    ],
    'Wrong_Damaged_Product': [
        "Product return kaise karu?",
        "Wrong item mila toh kya karu?",
        "Product exchange kaise karu?",
        "Refund kaise le sakte hai?",
        "Product ka warranty kya hai?",
        "Wrong item ka kya karu?",
        "Product return karne ka tarika kya hai?",
        "Product damage ho toh kya karu?",
        "Product exchange ke liye kya chahiye?",
        "Refund ke liye kya kya steps hai?",
        "Product return karne me kitna time lagta hai?",
        "Wrong item ka label kaise print karu?",
        "Product return karne ka option kahan hai?",
        "Refund ke liye kya kya documents chahiye?",
        "Product return karne ke baad kya hota hai?",
        "Wrong item ka status kaise check karu?",
        "Product return karne ka time limit kya hai?",
        "Refund process kaise hota hai?",
        "Product return karne ke liye kya kya chahiye?",
        "Wrong item ka refund kab aata hai?",
        "Mera order damage aaya hai, return kaise karu?",
        "Order me wrong product aaya, exchange kaise karu?",
        "Order delivered wrong item hai, refund kaise lunga?",
        "Mera order ka product kharab hai, return kaise karu?",
        "Order wrong color aaya, exchange ho sakta hai kya?",
        "Mera order incomplete hai, kya karu?",
        "Order ka product kharab hai, return option kahan hai?",
        "Mera order missing items hai, refund kaise lunga?",
        "Order wrong size aaya hai, exchange kaise karu?",
        "Mera order duplicate product aaya hai, kya karu?",
    ],
    'Customer_Service': [
        "Customer care ka number kya hai?",
        "Customer care kab tak available hai?",
        "Customer care se kaise baat karu?",
        "Customer care ka email kya hai?",
        "Customer care me complaint kaise karu?",
        "Customer care ka response time kitna hai?",
        "Customer care se escalation kaise karu?",
        "Customer care ka wait time kitna hai?",
        "Customer care kaise contact karu?",
        "Customer care ka process kya hai?",
        "Customer care me case kaise raise karu?",
        "Customer care ka timing kya hai?",
        "Customer care se help kaise le?",
        "Customer care ka escalation process kya hai?",
        "Customer care ka feedback kaise du?",
        "Customer care ka number change hua hai kya?",
        "Customer care ka support kya kya cover karta hai?",
        "Customer care me complaint status kaise check karu?",
        "Customer care ka resolution time kitna hai?",
        "Customer care ka contact options kya hai?",
    ],
    'Product_Quality': [
        "Product ka warranty period kya hai?",
        "Product quality kaise check karu?",
        "Product ka return policy kya hai?",
        "Product ka material kya hai?",
        "Product ki warranty claim kaise karu?",
        "Product ka quality certificate hai kya?",
        "Product ki durability kaise check karu?",
        "Product ka replacement policy kya hai?",
        "Product ki quality testing kaise hoti hai?",
        "Product ka manual kahan milega?",
        "Product ki specifications kya hai?",
        "Product ka user guide kahan hai?",
        "Product ki quality guarantee hai kya?",
        "Product ka after-sales service kya hai?",
        "Product ki quality report kahan hai?",
        "Product ka inspection kaise karu?",
        "Product ki quality standards kya hai?",
        "Product ka defect policy kya hai?",
        "Product ki quality assurance kya hai?",
        "Product ka maintenance guide kahan hai?",
    ],
    'Pricing_Discount': [
        "Coupon code kaise use karu?",
        "Discount kaise milega?",
        "Price match guarantee kya hai?",
        "Loyalty points kaise earn karu?",
        "EMI options kya hai?",
        "Cashback kaise milega?",
        "Promotional offer kab tak hai?",
        "Price drop ka notification kaise milega?",
        "Coupon code kahan se milega?",
        "Discount percentage kya hai?",
        "Price comparison kaise karu?",
        "Coupon code ka expiry kya hai?",
        "Price history kaise dekhu?",
        "Discount applicable kya hai?",
        "Price guarantee kya hai?",
        "Coupon code ka limit kya hai?",
        "Price tracking kaise karu?",
        "Discount coupon kahan milega?",
        "Price alert kaise set karu?",
        "Coupon code ka usage kya hai?",
    ],
}


# ============================================================================
# VOCABULARY
# ============================================================================

STATUS_WORDS = ['stuck', 'pending', 'not updating', 'failed', 'not working', 'blocked',
                'showing error', 'not responding', 'disabled', 'halted', 'stopped']

STATUS_WORDS2 = ['not confirmed', 'still pending', 'not placed', 'on hold']

URGENCY_WORDS_HIGH = ['bahut urgent hai', 'turant resolve karo', 'immediately', 'jaldi karo',
                      'abhi karo', 'fatafat', 'asap', 'very urgent', 'critical hai',
                      'emergency hai', 'right now', 'please jaldi', 'bahut der ho gayi']

URGENCY_WORDS_MEDIUM = ['please check karo', 'jaldi dekho', 'update do', 'kya problem hai',
                        'koi batao', 'help karo', 'please help', 'check karo']

URGENCY_WORDS_LOW = ['usually kitne din', 'kaise karu', 'kya hai', 'kahan hai',
                     'kya hota hai', 'kaise hota hai', 'kya process hai']

TIME_WORDS = ['2 din', '3 din', '5 din', '7 din', '10 din', '15 din', '1 hafta',
              '2 hafta', 'ek mahina', 'bahut din', 'kaafi din']

DAMAGE_WORDS = ['toot gaya', 'damage ho gaya', 'kharab hai', 'broken hai',
                'cracked hai', 'smashed', 'crushed', 'scratch aaya hai',
                'dent pada hai', 'bent ho gaya', 'phat gaya', 'missing hai']

PRODUCT_WORDS = ['phone', 'laptop', 'headphones', 'watch', 'camera', 'tablet',
                 'speaker', 'keyboard', 'mouse', 'charger', 'cable', 'cover']

PRODUCT_WORDS2 = ['different product', 'wrong item', 'completely different thing',
                  'kuch aur', 'something else']

COLOR_WORDS = ['red', 'blue', 'black', 'white', 'green', 'grey', 'pink']

SIZE_WORDS = ['M', 'L', 'XL', 'XXL', '42', '40', '38']
SIZE_WORDS2 = ['S', 'M', 'L', 'XL', '39', '41', '43']

ADDRESS_WORDS = ['sector 5, Noida', 'Andheri West, Mumbai', 'Koramangala, Bangalore',
                 'DLF Phase 5, Gurgaon', 'Whitefield, Bangalore']

AMOUNTS = ['500', '1000', '1500', '2000', '3000', '5000', '8000', '10000']
AMOUNTS2 = ['450', '900', '1400', '1800', '2700', '4500', '7500', '9000']

# ============================================================================
# SPELLING VARIATIONS
# ============================================================================

SPELLING_VARIANTS = {
    'nahi': ['nai', 'nahin', 'nahee'],
    'hai': ['he', 'hae'],
    'karo': ['kro', 'karoo'],
    'kab': ['kb'],
    'mera': ['meraa', 'mra'],
    'bahut': ['bht', 'bohot', 'bahot'],
    'jaldi': ['jldi', 'jaldee'],
    'abhi': ['abhee', 'abbi'],
    'mujhe': ['mujje', 'muje'],
    'bhi': ['bhee', 'bii'],
    'wapas': ['wapss', 'vapas'],
    'kyun': ['kyn', 'kyu'],
    'kya': ['kiya', 'kyaa'],
    'yeh': ['ye', 'yae'],
    'woh': ['vo', 'wo'],
    'se': ['sey'],
    'pe': ['pay'],
    'bheja': ['bhejaa', 'bhejha'],
    'refund': ['refnd', 'refound'],
    'delivery': ['delivry', 'delievery', 'delivr'],
    'order': ['ordr', 'oder'],
    'payment': ['paymnt', 'payement'],
    'return': ['retrn', 'reurn'],
    'product': ['prodct', 'prduct'],
    'account': ['acount', 'acconut'],
    'issue': ['isue', 'issu'],
    'please': ['plss', 'plz'],
    'urgent': ['urgnt', 'urgenttt'],
    'immediately': ['immeditely', 'immediatly'],
    'problem': ['problm', 'probelm'],
    'address': ['adress', 'adres'],
    'customer': ['custmer', 'custoemr'],
    'support': ['suport', 'supoort'],
    'package': ['parcl', 'pkg'],
    'track': ['trackng', 'trak'],
    'missing': ['msising', 'missng'],
    'damaged': ['dmaged', 'damaaged'],
    'wrong': ['wrng', 'wrrong'],
    'check': ['chek', 'chck'],
    'correct': ['corect', 'corrct'],
    'update': ['upadte', 'updte'],
}


def apply_spelling_variation(text, prob=0.3):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < prob and word.lower() in SPELLING_VARIANTS:
            variants = SPELLING_VARIANTS[word.lower()]
            new_words.append(random.choice(variants))
        else:
            new_words.append(word)
    return ' '.join(new_words)


def fill_template(template, urgency_level):
    """Fill a template with appropriate vocabulary based on urgency."""
    urgency_words = URGENCY_WORDS_HIGH if urgency_level == 'High' else \
                    URGENCY_WORDS_MEDIUM if urgency_level == 'Medium' else \
                    URGENCY_WORDS_LOW

    filled = template.format(
        status_word=random.choice(STATUS_WORDS),
        status_word2=random.choice(STATUS_WORDS2),
        urgency_word=random.choice(urgency_words),
        time_word=random.choice(TIME_WORDS),
        time_word2=random.choice(['pehle', 'kal', 'parson', 'is hafte']),
        damage_word=random.choice(DAMAGE_WORDS),
        product_word=random.choice(PRODUCT_WORDS),
        product_word2=random.choice(PRODUCT_WORDS2),
        color_word=random.choice(COLOR_WORDS),
        size_word=random.choice(SIZE_WORDS),
        size_word2=random.choice(SIZE_WORDS2),
        address_word=random.choice(ADDRESS_WORDS),
        amount=random.choice(AMOUNTS),
        amount2=random.choice(AMOUNTS2),
        num=random.randint(2, 30),
        num2=random.randint(1, 10),
    )
    return filled


def generate_complaint(category, urgency):
    """Generate a single synthetic complaint with urgency-appropriate tone."""
    if urgency == 'High':
        templates = HIGH_URGENCY_TEMPLATES[category]
    elif urgency == 'Medium':
        templates = MEDIUM_URGENCY_TEMPLATES[category]
    else:
        templates = LOW_URGENCY_TEMPLATES[category]

    template = random.choice(templates)
    text = fill_template(template, urgency)
    text = apply_spelling_variation(text, prob=0.2)
    return text


def augment_dataset(df, target_per_class=167):
    """Augment dataset with urgency-appropriate synthetic complaints."""
    augmented_rows = []

    # Calculate urgency distribution from original data per category
    urgency_dist = df.groupby('category')['urgency'].value_counts(normalize=True).to_dict()

    for category in df['category'].unique():
        existing_count = len(df[df['category'] == category])
        needed = target_per_class - existing_count

        if needed <= 0:
            continue

        print(f"  Generating {needed} samples for {category}...")

        # Get urgency distribution for this category
        dist = {urg: urgency_dist.get((category, urg), 1/3) for urg in ['High', 'Medium', 'Low']}

        for _ in range(needed):
            urgency = random.choices(
                list(dist.keys()),
                weights=list(dist.values())
            )[0]

            text = generate_complaint(category, urgency)
            augmented_rows.append({
                'text': text,
                'category': category,
                'urgency': urgency,
                'is_synthetic': True,
            })

    df_augmented = pd.DataFrame(augmented_rows)
    df_original = df.copy()
    df_original['is_synthetic'] = False

    df_combined = pd.concat([df_original, df_augmented], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_combined


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_ecommerce_complaints_360_spelling_variants.csv")

    print("Loading original dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Original: {len(df)} samples")

    print("\nAugmenting dataset...")
    df_augmented = augment_dataset(df, target_per_class=167)
    print(f"  Augmented: {len(df_augmented)} samples")

    output_path = os.path.join(PROJECT_ROOT, "data", "raw", "hinglish_complaints_augmented.csv")
    df_augmented.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")

    print(f"\n  Final distribution:")
    print(f"  Categories: {df_augmented['category'].value_counts().to_dict()}")
    print(f"  Urgency: {df_augmented['urgency'].value_counts().to_dict()}")
    print(f"  Synthetic: {df_augmented['is_synthetic'].sum()} new samples")
