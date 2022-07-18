source="https://arxiv.org/list/cs.{CC,FL,CL,DS}/pastweek?show=1000"

abstract() {
    while read code
    do
        abstract_ongoing='0'

        curl --silent "https://arxiv.org/abs/$code" |
        while read line
        do
            [[ $line == *"<span class=\"descriptor\">Abstract:</span>"* ]] && abstract_ongoing='1'
            [[ $line == *"</blockquote>"* ]] && abstract_ongoing='0'

            [ "$abstract_ongoing" -eq '1' ] && echo $line
        done
    done
}

curl --silent $source |
    grep '/abs/' |
    sed 's/.*\/abs\///' |
    sed 's/".*//' |
    abstract |
    sed 's/<span class=\"descriptor\">Abstract:<\/span> //'