SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/../downloads"

response=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/XPthmNk_pqEDaQ")
link=$(echo "$response" | jq -r '.href')

wget -O archive.zip "$link"
unzip -o archive.zip
rm archive.zip