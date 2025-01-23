
## Package versioning 
### Versioning system
This section is still under development. The versioning is based on https://py-pkgs.org/07-releasing-versioning.html

**Versioning steps**
1. make changes to package source files
2. bumb version number
3. run tests
4. update documentation
5. tag a release on version management (e.g., Git)
#### Version numbering
Everytime a change is made in the main versioning system branch the version number is changed.
- Patch release (0.1.0 -> 0.1.1): patch releases are typically used for bug fixes, which are backward compatible (changes from a hotfix branch or a release branch)
- Minor release (0.1.0 -> 0.2.0): a minor release typically includes larger bug fixes or new features that are backward compatible. (changes from a release branch)
- Major release (0.1.0 -> 1.0.0): release 1.0.0 is typically used for the first stable release of your package. After that, major releases are made for changes that are not backward compatible and may affect many users (changes from a release branch)
### Branching model
There is a set approach of how to update the package/program. This approach is based on a branching model. The branch with the code which the users use is called main (also refered to as production). There is a clear structure when things are allowed to be merged into main. 
1. NEVER push to main. 
2. Always first create and test the changes on a developer branch 
3. Only merge from a release or hotfix branch

The git branching model followed in this project is based on https://nvie.com/posts/a-successful-git-branching-model/.
#### `main` branch
There is a branch called `main`, this is the production branch. Which means this is the (only) branch users interact with. Any changes on this branch lead to a new version update. There are two ways that one can merge into  `main`
1. From a release branch.
2. From a hotfix branch.

Every time a branch is merged into main the main branched needs to be tagged with the version number
#### `develop` branch
Any changes ongoing changes are done in the `develop` branch. The following things can be done with the `develop` branch:
1. Direct pushes into develop with smaller changes and fixes
2. merging of feature branches, when the feature is finished, this is mostly larger changes which take time and there might be ongoing work on the develop branch at the same time
3. merging of hotfixes, everytime a hotfix is merged into the main branch, then the hotfix must also be merged into production
#### `feature` branch
feature branches are called `feature/{name of feature}` and is a branch of the `develop` branch. Information on feature branches:
- When a feature is finished it is merged into the develop branch and the feature branch is deleted
- A feature branch is mostly just stored locally, but can also be pusehd to the remote
- release bracnhes are called `release/{name of release}` and is a branch of the `develop` branch. 
#### `release` branch
Information on release branches:
- The name of the release is based on the continuous numbering and will either be 0.x or x.0, since it is about medium to larger versions
- In the release branch minor changes and fixes can be made before it is merged into `main`
- The release must be merged into the `develop` and `main`
- After the merge the release branch is deleted

How to make new release:
1. Create a new local release branch 
    ``` git checkout -b release-{release number} develop ```
    Make any final changes here and commit them, e.g., bumpbing release number
2. Check if the local branch is created
    ``` git branch -a ```
3. Push the local branch to the remote repository by setting the upstream branch 
    ``` git push -u origin release-{release number} ```
4. Check if the remote branch is created
    ``` git branch -r ```
5. If you are not tracking the main branch locally run:
    ``` git switch develop ```
6. Merge the release branch into the main branch 
    ``` git merge release-{release number} ```
7. Add a tag to the new release 
    ``` git tag -a v{release number} -m "{tag message}" ```
8. Push the changes to the remote main (production) branch
    ``` git push origin main ```  
9. Push the tag to the remote main (production) branch
    ``` git push origin main v{release number} ```
10. Switch back to the develop branch
    ``` git checkout develop ```
11. Merge any final changes frem the release branch 
    ``` git merge release-{release number} ```
12. Push the changes
    ``` git push origin develop ``` 
13. Delete the release branch locally
    ``` git branch -d release-{release number} ```
14. Detele the release branch remotely 
    ``` git push origin -d release-{release number} ```

#### `hotfix` branch
hotfix branches are called `hotfix/{name of hotfix}` and is a branch of the `main` branch. Information on hotfix branches:
- The name of the hotfix is based on the continuous numbering and will be a 0.0.x step
- the hotfix is merged with `main` and `develop`
- The hotfix is deleted after it has been merged
