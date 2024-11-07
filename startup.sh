if ! grep -q "deb http://ftp.debian.org/debian stable main" /etc/apt/sources.list; then
  echo "deb http://ftp.debian.org/debian stable main" >> /etc/apt/sources.list
fi
apt update -y && apt upgrade -y && apt install -y sqlite3