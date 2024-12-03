from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.padding import OAEP
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# 加载 RSA 私钥（从 PEM 格式字符串加载）
def load_private_key(pem_data):
    private_key = serialization.load_pem_private_key(
        pem_data.encode(),
        password=None,
        backend=default_backend()
    )
    return private_key

# 使用 RSA 私钥进行解密
def rsa_decrypt(private_key, encrypted_data):
    # 使用 OAEP 填充方式进行解密
    plaintext = private_key.decrypt(
        encrypted_data,
        OAEP(
            mgf=OAEP.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext

# 示例解密过程
if __name__ == "__main__":
    # 你提供的 RSA 私钥
    private_key_pem = """
    -----BEGIN PRIVATE KEY-----
    MIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDHJ3LfB58L5yNP
    uST7QNkacb/wV8OuWrWfIe95djmO0IyywdJg3LdD9r7IUjL4fu89vUZqZNjKx/A3
    kdq5zXSKw3pktjwVBBpc2OnhGv8SYClp0nLRbpCmARNpRz7dSYQmvF8vpmuAPHzt
    OP7PU+6nZBb6xkwr/8Qioa+SvElEicGmsy0+7xpMVKUX+BVOKWbxh7E8iflhT7d5
    ag9/7mFU4qa+ZKsiQgwRkg6D/bMlv/7p4PUR5rsNyXfVD2g01THL0TqOveK3n913
    KBl3Iolh9uObKk8l0ij5uVE0oeeJZ9X5XHp7tMJzpVf4cEp+MSMb9wH7VI4MopOH
    I0QIuq1zAgMBAAECggEAdIUYKHWFEKnDdzmT8Y0XeOlkq3IuAyz/ZoOsYRxSwMQ0
    DcJpHFMGxrTvGrU9LTbXMwAy2rz2Om6QlXK4zkzvCuEkExisPn+QDRK8hAAPjOjG
    UivXEcHmz7mBae9NUJtavm8oIfD0pKq/TOwz6Ynp7/YXm9G5b2TNlJWU6/1NwNLu
    7hBNIGOohr2q4EkOjyF1fxxTmfHD+jJlTaogYXSFqj8LskZIygFE9QRsAkk41KJR
    v3+9aeRxl4zs8ifdoo8B7l7w2YpGtF2leHJa+rrkfEedWqsMRuyGzX+WeO2xRPWP
    AC9Hh7jPeo5ON3ixFqjGPlIh0851VXT9EqAfJkNwwQKBgQD56TRwkJ/nWoGmlE1L
    JsOUdiWpOpwEer4dIfZvrZs+QNmktoqzfoyp3RV4qh45d0gcGXPedGeDAgnWyBBL
    96/jfRh8yBz2KgmJ4SFge37asqQkGWFuIFLGNX4uGynWnsy/FnDCVDm0Jj2rIbIs
    d0+MR6Tkur+YgwWsCmtqCevAVwKBgQDMAacl5XpZYnsD3n5OSsF/lX0lZS9mNnuZ
    zp2sB2H9wIWhS0vtK0CiM/8rQDozPxW6GkiuuDCuYOVcXryrC1XhUA0VNPSQey9I
    s/ANqMJmKG65gnnxudyoMAaNz8mA5u2i/8Y6AOpexUZJuPJL26i6FjpKuA+VbLOx
    prfW9jEaRQKBgQDtLax9IGUB9t2RMLJinnmDztVTVLJ5ddw0XeU6fDMX1Ag60Ju2
    WmY5V/9ms11YAKLJOEbFWwhaR3b7Boig8INXjYPN+UWzQpYm6yj4Hnx4Jo6tTAEx
    uS+VuXL1YwZEEBYVTMDbTYAuPxTL84DbvqgaZGxUQABSSBb7/i+PRbcepQKBgQCA
    8+KGD+IgsiF0NqW8M4DQdtveUXF+uJ20gWglH52PWqydYg0iY569aQS4gCbJ0eyX
    8JlU59TNxS32D2RO8iFdBM7gQtL8qQEgga0R1UTccl5bIOCYLZYPMhxSc6+5rT81
    M1xHueBr+2MMor11uemThw1dwa8IEugbOXknhgNPyQKBgQDCzQgvEhELBYYPUl85
    Yby+WQUvHwDJkfU0Oy9n6Lbd5LDbgxSzNPYIJgeZyyhjcoAtX9ar3bg+hUXLtnmJ
    WYdr8Um/JpNO8bootfUW9zOg0RgNoTpjtoBGMmiMnk8D86B0sNugJev+/ptPC24o
    nDqTbaWqnUAeV/l5tC3SCDWwVQ==
    -----END PRIVATE KEY-----
    """

    # 加载私钥
    private_key = load_private_key(private_key_pem)

    # 假设这是通过公钥加密后的数据（需要你替换为实际的加密数据）
    encrypted_data = b'l4NBDIgpUA5XGsjlfm7OJBlHbOcfdUSp-Kw7pDGiPgVU9UxQqCzLXqArtaHd17uE3jQhRlL1T46g1hoRNovGh1cdjlXvbXDfK324cArG15VIY0cys6z69F9Fl_J7P9PhOcHe9k8Cf7Rh8L0K6q-Y2Ny9bT7Hr2KAnDfrH8bR8BdbqTxdcK2Egv31xiEXp_YFphFUrjcH7HHrNkgjD_N0kvTqexuExSv7hMzW4b8RB7MdpnTBF9Ot2LrDTd-zxwsLbUltr_ihr6fvlX9Ydrg_s6JAjaswv280rwR6lR3PMsrG5pXGpEXVbwtHUqkwJZtNpuIyXM6mz7Yu_NEctYJHkQ=='  # 这里需要是实际的加密数据

    # 解密
    decrypted_data = rsa_decrypt(private_key, encrypted_data)
    print("Decrypted data:", decrypted_data.decode())
