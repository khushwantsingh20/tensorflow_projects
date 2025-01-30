# import itertools

# def brute_force_password(target_password):
#     for attempt in itertools.product('0123456789', repeat=4):  # Generates all 4-digit combinations
#         attempt_password = ''.join(attempt)
#         print(f"Trying: {attempt_password}")
#         if attempt_password == target_password:
#             print(f"Password found: {attempt_password}")
#             return attempt_password
#     print("Password not found")
#     return None

# if __name__ == "__main__":
#     password = "1254"  # Set your 4-digit password
#     brute_force_password(password)



import itertools

def brute_force_password(target_password, charset):
    attempt_count = 0
    for attempt in itertools.product(charset, repeat=len(target_password)):
        attempt_password = ''.join(attempt)
        attempt_count += 1
        print(f"Trying: {attempt_password} (Attempt: {attempt_count})")
        if attempt_password == target_password:
            print(f"Password found: {attempt_password} in {attempt_count} attempts")
            return attempt_password, attempt_count
    print("Password not found")
    return None, attempt_count

if __name__ == "__main__":
    password = "1234"  # Set your password with numbers, chars, and special chars
    charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"  # Define possible characters
    brute_force_password(password, charset)
