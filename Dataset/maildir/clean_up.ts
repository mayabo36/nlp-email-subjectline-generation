const fs = require('fs');
const path = require('path');

fs.readdirSync(
    path.dirname(
        __filename
    ))
    .filter(x => !x.endsWith('.ts'))
    .forEach(x => {

        fs.readdir(__dirname + '/' + x + '/', (err, files) => {
            let xs = (files || []).reduce((acc, file) => {
                if (file !== 'sent' && file !== 'inbox' && file !== 'sent_items')
                    acc.push(x + '/' + file);
                return acc;
            }, []);
            if (xs.length)
                console.log(`rm -rf ${xs.join(' ')} ;`);
            else
                console.log(`${x} had no folders to remove.`)
        });
    });